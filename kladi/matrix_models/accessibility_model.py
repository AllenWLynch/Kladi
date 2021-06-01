
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints
from tqdm import tqdm
import warnings
from sklearn.preprocessing import scale
from scipy import sparse
from scipy.stats import fisher_exact
from scipy.sparse import isspmatrix
from kladi.matrix_models.scipm_base import BaseModel
from kladi.motif_scanning import moods_scan
import configparser
import requests
import json
from lisa.core.utils import indices_list_to_sparse_array
import logging


class DANEncoder(nn.Module):

    def __init__(self, num_peaks, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_peaks, hidden, padding_idx=0)
        self.fc1 = nn.Linear(hidden + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse

    def forward(self, idx, read_depth):

        embeddings = self.drop(self.embedding(idx)) # N, T, D
        
        ave_embeddings = embeddings.sum(1)/read_depth

        h = torch.cat([ave_embeddings, read_depth.log()], dim = 1) #inject read depth into model
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.drop2(h)

        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = F.softplus(theta_scale)  # Enforces positivity
        return theta_loc, theta_scale

class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, num_genes, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_genes, bias = False)
        self.bn = nn.BatchNorm1d(num_genes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class AccessibilityModel(BaseModel):

    def __init__(self, peaks, num_modules = 15, initial_counts = 10, 
        dropout = 0.2, hidden = 128, use_cuda = True):

        assert(isinstance(peaks, (list, np.ndarray)))
        assert(len(peaks.shape) == 2)
        
        self.num_features = len(peaks)
        self.peaks = np.array(peaks)
        super().__init__(self.num_features, 
            DANEncoder(self.num_features, num_modules, hidden, dropout), 
            Decoder(self.num_features, num_modules, dropout), 
            num_topics = num_modules, initial_counts = initial_counts, 
            hidden = hidden, dropout = dropout, use_cuda = use_cuda)


    def model(self, peak_idx, read_depth, onehot_obs = None):

        pyro.module("decoder", self.decoder)
        
        #pyro.module("decoder", self.decoder)
        with pyro.plate("cells", peak_idx.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution

            theta_loc = self.prior_mu * peak_idx.new_ones((peak_idx.shape[0], self.num_topics))
            theta_scale = self.prior_std * peak_idx.new_ones((peak_idx.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            peak_probs = self.decoder(theta)
            
            pyro.sample(
                'obs',
                dist.Multinomial(total_count = read_depth if onehot_obs is None else 1, probs = peak_probs).to_event(1),
                obs=onehot_obs
            )

    def guide(self, peak_idx, read_depth, onehot_obs = None):

        pyro.module("encoder", self.encoder)

        with pyro.plate("cells", peak_idx.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(peak_idx, read_depth)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))


    def _get_latent_MAP(self, idx, read_depth, oneshot_obs = None):
        theta_loc, theta_scale = self.encoder(idx, read_depth)
        
        Z = theta_loc.cpu().detach().numpy()
        return np.exp(Z)/np.exp(Z).sum(-1, keepdims = True)

    def _validate_data(self, X):
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if not isspmatrix(X):
            X = sparse.csr_matrix(X)

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_features)
        
        assert(np.isclose(X.data.astype(np.int64), X.data).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        logging.info('Binarizing accessibility matrix ...')
        X.data = np.ones_like(X.data)

        return X


    def _get_onehot_tensor(self, idx):
        return torch.zeros(idx.shape[0], self.num_features, device = self.device).scatter_(1, idx, 1).to(self.device)

    def _get_padded_idx_matrix(self, accessibility_matrix, read_depth):

        width = read_depth.max()

        dense_matrix = []
        for i in range(accessibility_matrix.shape[0]):
            row = accessibility_matrix[i,:].indices
            if len(row) == width:
                dense_matrix.append(np.array(row)[np.newaxis, :])
            else:
                dense_matrix.append(np.concatenate([np.array(row), np.zeros(width - len(row))])[np.newaxis, :]) #0-pad tail to "width"

        dense_matrix = np.vstack(dense_matrix)
        
        return dense_matrix.astype(np.int64)


    def _get_batches(self, accessibility_matrix, batch_size = 32, training = True):

        N = accessibility_matrix.shape[0]

        assert(isspmatrix(accessibility_matrix))
        assert(accessibility_matrix.shape[1] <= self.num_features)

        accessibility_matrix = accessibility_matrix.tocsr()
        read_depth = torch.from_numpy(np.array(accessibility_matrix.sum(-1))).to(self.device)

        for batch_start, batch_end in self._iterate_batch_idx(N, batch_size, bar = True):

            rd_batch = read_depth[batch_start:batch_end]
            idx_batch = torch.from_numpy(self._get_padded_idx_matrix(accessibility_matrix[batch_start : batch_end], rd_batch)).to(self.device)

            onehot_batch = self._get_onehot_tensor(idx_batch) if training else None

            yield idx_batch, read_depth[batch_start:batch_end], onehot_batch

    def save(self, filename):
        state_dict = dict(state = self.state_dict())
        state_dict['peaks'] = self.peaks
        
        try:
            self.motif_hits
        except AttributeError:
            pass
        else:
            motif_dictionary = dict(
                indptr = self.motif_hits.indptr,
                indices = self.motif_hits.indices,
                score = self.motif_hits.data,
                ids = self.motif_ids,
                factor_names = self.factor_names,
            )
            state_dict['motifs'] = motif_dictionary

        torch.save(state_dict, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state'])
        self.eval()
        self.peaks = state['peaks']
        self.set_device('cpu')

        if 'motifs' in state:
            self.motif_ids = state['motifs']['ids']
            self.factor_names = state['motifs']['factor_names']
            self.motif_hits = sparse.csr_matrix((state['motifs']['score'], state['motifs']['indices'], state['motifs']['indptr']), shape = (len(self.motif_ids), len(self.peaks)))


    def _argsort_peaks(self, module_num):
        assert(isinstance(module_num, int) and module_num < self.num_topics and module_num >= 0)
        return np.argsort(self._get_beta()[module_num, :])

    def rank_peaks(self, module_num):
        return self.peaks[self._argsort_peaks(module_num)]

    def get_top_peaks(self, module_num, top_n = 20000):
        return self.rank_peaks(module_num)[-top_n : ]

    def _batch_impute(self, latent_compositions, batch_size = 256):

        self._check_latent_vars(latent_compositions)

        latent_compositions = self._to_tensor(latent_compositions)
        logging.info('Finding posterior peak probabilities ...')
        for batch_start, batch_end in self._iterate_batch_idx(len(latent_compositions), batch_size, bar = True):
            yield self.decoder(latent_compositions[batch_start:batch_end, :]).cpu().detach().numpy()

    def _validate_hits_matrix(self, hits_matrix):
        assert(isspmatrix(hits_matrix))
        assert(len(hits_matrix.shape) == 2)
        assert(hits_matrix.shape[1] == len(self.peaks))
        hits_matrix = hits_matrix.tocsr()

        hits_matrix.data = np.ones_like(hits_matrix.data)
        return hits_matrix


    def get_motif_hits_in_peaks(self, genome_fasta, p_value_threshold = 0.00005):
        
        self.motif_hits, self.motif_ids, self.factor_names = moods_scan.get_motif_enrichments(self.peaks.tolist(), genome_fasta, pvalue_threshold = p_value_threshold)


    def get_motif_score(self, latent_compositions):

        try:
            self.motif_hits
        except AttributeError:
            raise Exception('Motif hits not yet calculated, run "get_motif_hits_in_peaks" function first!')

        hits_matrix = self._validate_hits_matrix(self.motif_hits)

        motif_scores = np.vstack([hits_matrix.dot(np.log(peak_probabilities).T).T
            for peak_probabilities in self._batch_impute(latent_compositions)])

        motif_scores = scale(motif_scores/np.linalg.norm(motif_scores, axis = 1, keepdims = True))

        return list(zip(self.motif_ids, self.factor_names)), motif_scores


    def enrich_TFs(self, module_num, top_quantile = 0.2):

        try:
            self.motif_hits
        except AttributeError:
            raise Exception('Motif hits not yet calculated, run "get_motif_hits_in_peaks" function first!')

        hits_matrix = self._validate_hits_matrix(self.motif_hits)
        assert(isinstance(top_quantile, float) and top_quantile > 0 and top_quantile < 1)

        module_idx = self._argsort_peaks(module_num)[-int(self.num_features*top_quantile) : ]
        logging.info('Finding enrichment in top {} peaks ...'.format(str(len(module_idx))))

        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0])):

            tf_hits = hits_matrix[i,:].indices
            overlap = len(np.intersect1d(tf_hits, module_idx))
            module_only = len(module_idx) - overlap
            tf_only = len(tf_hits) - overlap
            neither = self.num_features - (overlap + module_only + tf_only)

            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            test_statistics.append(stat)

        return sorted(list(zip(self.motif_ids, self.factor_names, pvals, test_statistics)), key = lambda x:(x[2], -x[3]))


    def _insilico_deletion(self, gene_models, deletion_masks, latent_compositions):

        assert(sparse.isspmatrix(deletion_masks))
        deletion_masks = deletion_masks.tocoo().astype(bool).astype(int)
        assert(deletion_masks.shape[1] == len(self.peaks))
        deletion_masks.eliminate_zeros()
        
        num_states = len(latent_compositions)
        num_genes = len(gene_models)
        num_knockouts = deletion_masks.shape[0]
        num_steps = num_states * num_genes

        reg_states = list(self._batch_impute(latent_compositions, batch_size = num_states))[0]

        del_i, del_j = deletion_masks.row, deletion_masks.col

        logging.info('Calculating insilico-deletion scores ...')
        with tqdm(total=num_steps) as bar:
            
            state_knockout_scores = []
            for reg_state in reg_states:
                
                reg_state = reg_state[np.newaxis, :]
                knockout_states = np.repeat(reg_state, num_knockouts, axis = 0)
                knockout_states[del_i, del_j] = 0

                isd_states = np.vstack([reg_state, knockout_states])

                rp_scores = []
                for model in gene_models:
                    rp_scores.append(np.exp(model.predict(isd_states)))
                    bar.update(1)

                rp_scores = np.hstack(rp_scores)
                #num_knockouts + 1, num_genes
                rp_scores = 1 - rp_scores[1:]/rp_scores[0]
                state_knockout_scores.append(rp_scores.T[:,:, np.newaxis])

        state_knockout_scores = np.concatenate(state_knockout_scores, axis = -1)
        
        return state_knockout_scores

    def get_most_influenced_genes(self, rp_models, module_num, top_n_genes = 200, top_peak_quantile = 0.2):
        
        assert(isinstance(module_num, int) and module_num >= 0 and module_num < self.num_topics)
        assert(isinstance(top_peak_quantile, float) and top_peak_quantile > 0 and top_peak_quantile < 1.)
        assert(isinstance(top_n_genes, int) and top_n_genes > 0)

        module_idx = self._argsort_peaks(module_num)[-int(self.num_features*top_peak_quantile) : ]
        deletion_mask = indices_list_to_sparse_array([module_idx], self.num_features)

        reg_state = np.zeros(self.num_topics)[np.newaxis,:]
        reg_state[0,module_num] = 1.0
        reg_state = reg_state.astype(np.float32)

        isd_scores = np.ravel(self._insilico_deletion(rp_models, deletion_mask, reg_state))
        genes = [model.name for model in rp_models]

        sorted_gene_scores = sorted(zip(genes, isd_scores), key = lambda x : x[1])

        sorted_gene_symbols = list(zip(*sorted_gene_scores[-top_n_genes:]))[0]

        return sorted_gene_symbols