
import numpy as np
from numpy.core.numeric import ones
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
from kladi.matrix_models.scipm_base import BaseModel, get_fc_stack
from kladi.motif_scanning import moods_scan
from lisa.core.utils import indices_list_to_sparse_array
import logging
from lisa import FromRegions
import re


class ZeroPaddedBinaryMultinomial(pyro.distributions.Multinomial):
    
    def log_prob(self, value):
        if self._validate_args:
            pass
        #self._validate_sample(value)
        logits = self.logits
        logits = logits.clone(memory_format=torch.contiguous_format)
        
        log_factorial_n = torch.lgamma((value > 0).sum(-1) + 1)
        
        logits = torch.hstack([value.new_zeros((logits.shape[0], 1)), logits])

        log_powers = torch.gather(logits, -1, value).sum(-1)
        return log_factorial_n + log_powers


class DANEncoder(nn.Module):

    def __init__(self, num_peaks, num_topics, hidden, dropout, num_layers):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_peaks + 1, hidden, padding_idx=0)
        self.num_topics = num_topics
        self.fc_layers = get_fc_stack(
            layer_dims = [hidden + 1, *[hidden]*(num_layers-2), 2*num_topics],
            dropout = dropout, skip_nonlin = True
        )

    def forward(self, idx, read_depth):

        embeddings = self.drop(self.embedding(idx)) # N, T, D
        
        ave_embeddings = embeddings.sum(1)/read_depth

        X = torch.cat([ave_embeddings, read_depth.log()], dim = 1) #inject read depth into model

        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])   

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

    def __init__(self, peaks, num_modules = 15, num_layers = 3, 
        encoder_dropout = 0.1, decoder_dropout = 0.2, hidden = 128, use_cuda = True, highly_variable = None,
        seed = None):

        assert(isinstance(peaks, (list, np.ndarray)))
        if isinstance(peaks, list):
            peaks = np.array(peaks)
        
        if len(peaks.shape) == 1 and isinstance(peaks[0], str):
            peaks = np.array(list(
                map(lambda x : re.split(':|-', x), peaks)
            ))
        else:
            assert(len(peaks.shape) == 2)
            assert(peaks.shape[-1] == 3)
            assert(isinstance(peaks[0,0], str))
        
        self.peaks = peaks
        self.num_peaks = len(peaks) 
        
        kwargs = dict(
            num_modules = num_modules,
            num_exog_features = len(self.peaks),
            highly_variable = highly_variable,
            hidden = hidden,
            num_layers = num_layers,
            decoder_dropout = decoder_dropout,
            encoder_dropout = encoder_dropout,
            use_cuda = use_cuda,
            seed = seed,
        )

        super().__init__(DANEncoder, Decoder, **kwargs)
            

    def model(self, endog_peaks, exog_peaks, read_depth):

        pyro.module("decoder", self.decoder)

        _alpha, _beta = self._get_gamma_parameters(self.I, self.num_topics)
        with pyro.plate("topics", self.num_topics):
            initial_counts = pyro.sample("a", dist.Gamma(self._to_tensor(_alpha), self._to_tensor(_beta)))

        theta_loc = self._get_prior_mu(initial_counts, self.K)
        theta_scale = self._get_prior_std(initial_counts, self.K)
        
        #pyro.module("decoder", self.decoder)
        with pyro.plate("cells", exog_peaks.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            peak_probs = self.decoder(theta)
            
            pyro.sample(
                'obs',
                ZeroPaddedBinaryMultinomial(total_count = 1, probs = peak_probs),
                obs = exog_peaks,
            )
        
        return peak_probs

    def guide(self, endog_peaks, exog_peaks, read_depth):

        pyro.module("encoder", self.encoder)

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(*self._get_gamma_moments(self.I, self.num_topics))
        counts_mu = pyro.param('counts_mu', _counts_mu * read_depth.new_ones((self.num_topics,))).to(self.device)
        counts_std = pyro.param('counts_std', np.sqrt(_counts_var) * read_depth.new_ones((self.num_topics,)), 
                constraint = constraints.positive).to(self.device)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(counts_mu[k], counts_std[k]))

        with pyro.plate("cells", endog_peaks.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(endog_peaks, read_depth)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))


    def _get_latent_MAP(self, endog_peaks, exog_peaks, read_depth):
        theta_loc, theta_scale = self.encoder(endog_peaks, read_depth)
        
        Z = theta_loc.cpu().detach().numpy()
        return np.exp(Z)/np.exp(Z).sum(-1, keepdims = True)

    def _validate_data(self, X):
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if not isspmatrix(X):
            X = sparse.csr_matrix(X)

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_peaks)
        
        assert(np.isclose(X.data.astype(np.int64), X.data).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        logging.debug('Binarizing accessibility matrix ...')
        X.data = np.ones_like(X.data)

        return X


    def _get_padded_idx_matrix(self, accessibility_matrix):

        width = int(accessibility_matrix.sum(-1).max())

        dense_matrix = []
        for i in range(accessibility_matrix.shape[0]):
            row = accessibility_matrix[i,:].indices + 1
            if len(row) == width:
                dense_matrix.append(np.array(row)[np.newaxis, :])
            else:
                dense_matrix.append(np.concatenate([np.array(row), np.zeros(width - len(row))])[np.newaxis, :]) #0-pad tail to "width"

        dense_matrix = np.vstack(dense_matrix)
        
        return dense_matrix.astype(np.int64)


    def _get_batches(self, accessibility_matrix, batch_size = 32, bar = False, training = True):

        N = accessibility_matrix.shape[0]

        assert(isspmatrix(accessibility_matrix))
        assert(accessibility_matrix.shape[1] <= self.num_peaks)

        accessibility_matrix = accessibility_matrix.tocsr()
        read_depth = torch.from_numpy(np.array(accessibility_matrix.sum(-1))).to(self.device).type(torch.int32)

        for batch_start, batch_end in self._iterate_batch_idx(N, batch_size, bar = bar):

            rd_batch = read_depth[batch_start:batch_end]
            reads = accessibility_matrix[batch_start : batch_end]
            endog_peaks = torch.from_numpy(self._get_padded_idx_matrix(reads[:, self.highly_variable])).to(self.device)
            exog_peaks = torch.from_numpy(self._get_padded_idx_matrix(reads)).to(self.device)

            yield endog_peaks, exog_peaks, rd_batch


    def _get_save_data(self):

        data = super()._get_save_data()
        
        try:
            self.factor_hits
        except AttributeError:
            pass
        else:
            motif_dictionary = dict(
                indptr = self.factor_hits.indptr,
                indices = self.factor_hits.indices,
                score = self.factor_hits.data,
                ids = self.factor_ids,
                factor_names = self.factor_names,
            )
            data['motifs'] = motif_dictionary

        return data

    def _load_save_data(self, data):
        super()._load_save_data(data)

        if 'motifs' in data:
            self.factor_ids = data['motifs']['ids']
            self.factor_names = data['motifs']['factor_names']
            self.factor_hits = sparse.csr_matrix(
                (data['motifs']['score'], data['motifs']['indices'], data['motifs']['indptr']), 
                shape = (len(self.factor_ids), len(self.peaks))
            )


    def _argsort_peaks(self, module_num):
        assert(isinstance(module_num, int) and module_num < self.num_topics and module_num >= 0)
        return np.argsort(self._score_feature()[module_num, :])

    def rank_peaks(self, module_num):
        return self.peaks[self._argsort_peaks(module_num)]

    def get_top_peaks(self, module_num, top_n = 20000):
        return self.rank_peaks(module_num)[-top_n : ]

    def _batch_impute(self, latent_compositions, batch_size = 256):

        self._check_latent_vars(latent_compositions)

        latent_compositions = self._to_tensor(latent_compositions)
        logging.debug('Finding posterior peak probabilities ...')
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
        
        self.factor_hits, self.factor_ids, self.factor_names = moods_scan.get_motif_enrichments(self.peaks.tolist(), genome_fasta, pvalue_threshold = p_value_threshold)

    def get_ChIP_hits_in_peaks(self, species):

        regions_test = FromRegions(species, self.peaks.tolist())

        chip_hits, sample_ids, metadata = regions_test._load_factor_binding_data()

        bin_map = np.hstack([np.arange(chip_hits.shape[0])[:,np.newaxis], regions_test.region_score_map[:, np.newaxis]])

        new_hits = regions_test.data_interface.project_sparse_matrix(chip_hits, bin_map, num_bins=len(self.peaks))

        self.factor_hits = new_hits.T.tocsr()
        self.factor_names = list(metadata['factor'])
        self.factor_ids = list(sample_ids)


    def get_motif_score(self, latent_compositions):

        try:
            self.factor_hits
        except AttributeError:
            raise Exception('Motif hits not yet calculated, run "get_motif_hits_in_peaks" function first!')

        hits_matrix = self._validate_hits_matrix(self.factor_hits)

        motif_scores = np.vstack([hits_matrix.dot(np.log(peak_probabilities).T).T
            for peak_probabilities in self._batch_impute(latent_compositions)])

        motif_scores = scale(motif_scores/np.linalg.norm(motif_scores, axis = 1, keepdims = True))

        return list(zip(self.factor_ids, self.factor_names)), motif_scores


    def enrich_TFs(self, module_num, top_quantile = 0.2):

        try:
            self.factor_hits
        except AttributeError:
            raise Exception('Motif hits not yet calculated, run "get_motif_hits_in_peaks" function first!')

        hits_matrix = self._validate_hits_matrix(self.factor_hits)
        assert(isinstance(top_quantile, float) and top_quantile > 0 and top_quantile < 1)

        module_idx = self._argsort_peaks(module_num)[-int(self.num_peaks*top_quantile) : ]
        logging.info('Finding enrichment in top {} peaks ...'.format(str(len(module_idx))))

        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0])):

            tf_hits = hits_matrix[i,:].indices
            overlap = len(np.intersect1d(tf_hits, module_idx))
            module_only = len(module_idx) - overlap
            tf_only = len(tf_hits) - overlap
            neither = self.num_peaks - (overlap + module_only + tf_only)

            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            test_statistics.append(stat)

        return sorted(list(zip(self.factor_ids, self.factor_names, pvals, test_statistics)), key = lambda x:(x[2], -x[3]))


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

        module_idx = self._argsort_peaks(module_num)[-int(self.num_peaks*top_peak_quantile) : ]
        deletion_mask = indices_list_to_sparse_array([module_idx], self.num_peaks)

        reg_state = np.zeros(self.num_topics)[np.newaxis,:]
        reg_state[0,module_num] = 1.0
        reg_state = reg_state.astype(np.float32)

        isd_scores = np.ravel(self._insilico_deletion(rp_models, deletion_mask, reg_state))
        genes = [model.name for model in rp_models]

        sorted_gene_scores = sorted(zip(genes, isd_scores), key = lambda x : x[1])

        sorted_gene_symbols = list(zip(*sorted_gene_scores[-top_n_genes:]))[0]

        return sorted_gene_symbols