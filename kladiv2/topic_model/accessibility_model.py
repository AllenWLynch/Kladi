
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
from kladiv2.topic_model.base import BaseModel, get_fc_stack
from kladi.core.plot_utils import plot_factor_influence
import matplotlib.pyplot as plt
from pyro.contrib.autoname import scope
from pyro import poutine
from kladiv2.core.adata_interface import *
from kladiv2.plots.factor_influence_plot import plot_factor_influence
from sklearn.preprocessing import scale

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

    def __init__(self,*,num_endog_features, num_topics, hidden, dropout, num_layers):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_endog_features + 1, hidden, padding_idx=0)
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

    def topic_comps(self, idx, read_depth):
        theta = self.forward(idx, read_depth)[0]
        theta = theta.exp()/theta.exp().sum(-1, keepdim = True)
        
        return theta.detach().cpu().numpy()


class AccessibilityModel(BaseModel):

    encoder_model = DANEncoder

    @classmethod
    def load_old_model(cls, filename, varnames):

        old_model = torch.load(filename)

        params = old_model['params'].copy()
        params['num_topics'] = params.pop('num_modules')
        
        fit_params = {}
        fit_params['num_exog_features'] = len(params['features'])
        fit_params['num_endog_features'] = params['highly_variable'].sum()
        fit_params['highly_variable'] = params.pop('highly_variable')
        fit_params['features'] = varnames
        params.pop('features')
        
        model = cls(**params)
        model._set_weights(fit_params, old_model['model']['weights'])

        if 'pi' in old_model['model']:
            model.residual_pi = old_model['model']['pi']
        
        return model

    @property
    def peaks(self):
        return self.features
            
    @scope(prefix='atac')
    def model(self,*, endog_features, exog_features, read_depth, anneal_factor = 1.):
        theta_loc, theta_scale = super().model()
        
        with pyro.plate("cells", endog_features.shape[0]):

            with poutine.scale(None, anneal_factor):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                )

            theta = theta/theta.sum(-1, keepdim = True)
            
            peak_probs = self.decoder(theta)
            
            pyro.sample(
                'obs', ZeroPaddedBinaryMultinomial(total_count = 1, probs = peak_probs), obs = exog_features,
            )

    @scope(prefix = 'atac')
    def guide(self, *, endog_features, exog_features, read_depth, anneal_factor = 1.):
        super().guide()

        with pyro.plate("cells", endog_features.shape[0]):
            
            theta_loc, theta_scale = self.encoder(endog_features, read_depth)

            with poutine.scale(None, anneal_factor):
                    
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                )

    @staticmethod
    def _binarize_matrix(X, expected_width):
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if not isspmatrix(X):
            X = sparse.csr_matrix(X)

        assert(len(X.shape) == 2)
        assert(X.shape[1] == expected_width)
        
        assert(np.isclose(X.data.astype(np.int64), X.data, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

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

    
    def _preprocess_endog(self, X, read_depth):
        
        return torch.tensor(
            self._get_padded_idx_matrix(self._binarize_matrix(X, self.num_endog_features)), 
            requires_grad = False
        ).to(self.device)


    def _preprocess_exog(self, X):
        
        return torch.tensor(
            self._get_padded_idx_matrix(self._binarize_matrix(X, self.num_exog_features)), 
            requires_grad = False
        ).to(self.device)


    def _argsort_peaks(self, module_num):
        assert(isinstance(module_num, int) and module_num < self.num_topics and module_num >= 0)
        return np.argsort(self._score_features()[module_num, :])


    def rank_peaks(self, module_num):
        return self.peaks[self._argsort_peaks(module_num)]

    def _validate_hits_matrix(self, hits_matrix):
        assert(isspmatrix(hits_matrix))
        assert(len(hits_matrix.shape) == 2)
        assert(hits_matrix.shape[1] == len(self.peaks))
        hits_matrix = hits_matrix.tocsr()

        hits_matrix.data = np.ones_like(hits_matrix.data)
        return hits_matrix
    
    @wraps_modelfunc(adata_extractor = get_factor_hits, adata_adder = return_output,
        del_kwargs = ['hits_matrix','metadata'])
    def get_enriched_TFs(self, factor_type = 'motifs', top_quantile = 0.1, *, 
            module_num, hits_matrix, metadata):

        assert(isinstance(top_quantile, float) and top_quantile > 0 and top_quantile < 1)
        hits_matrix = self._validate_hits_matrix(hits_matrix)

        module_idx = self._argsort_peaks(module_num)[-int(self.num_exog_features*top_quantile) : ]

        pvals, test_statistics = [], []
        for i in tqdm(range(hits_matrix.shape[0]), 'Finding enrichments'):

            tf_hits = hits_matrix[i,:].indices
            overlap = len(np.intersect1d(tf_hits, module_idx))
            module_only = len(module_idx) - overlap
            tf_only = len(tf_hits) - overlap
            neither = self.num_exog_features - (overlap + module_only + tf_only)

            contingency_matrix = np.array([[overlap, module_only], [tf_only, neither]])
            stat,pval = fisher_exact(contingency_matrix, alternative='greater')
            pvals.append(pval)
            test_statistics.append(stat)

        results = [
            dict(**meta, pval = pval, test_statistic = test_stat)
            for meta, pval, test_stat in zip(metadata, pvals, test_statistics)
        ]

        self.enrichments[(factor_type, module_num)] = results

        return results


    @wraps_modelfunc(adata_extractor = get_factor_hits_and_latent_comps, adata_adder = get_motif_score_adata,
        del_kwargs = ['metadata','hits_matrix','topic_compositions'])
    def get_motif_scores(self, batch_size=512,*, metadata, hits_matrix, topic_compositions):

        hits_matrix = self._validate_hits_matrix(hits_matrix)
    
        motif_scores = np.vstack([
            hits_matrix.dot(np.log(peak_probabilities).T).T
            for peak_probabilities in self._batched_impute(topic_compositions)
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normalized_scores = scale(motif_scores/np.linalg.norm(motif_scores, axis=-1, keepdims=True))

        return metadata, motif_scores, normalized_scores


    def get_enrichments(self, module_num, factor_type = 'motifs'):
        try:
            return self.enrichments[(factor_type, module_num)]
        except KeyError:
            raise KeyError('User has not gotten enrichments yet for module {} using factor_type: {}. Run "get_enriched_TFs" function.'\
                .format(str(module_num), str(factor_type)))


    @staticmethod
    def join_factor_meta(m1, m2, show_factor_ids = False):

        def reformat_meta(meta):
            return {factor['id'] : factor for factor in meta}

        m1 = reformat_meta(m1)
        m2 = reformat_meta(m2)

        shared_factors = np.intersect1d(list(m1.keys()), list(m2.keys()))

        m1 = [m1[factor_id] for factor_id in shared_factors]
        m2 = [m2[factor_id] for factor_id in shared_factors]

        l1_pvals = np.array([x['pval'] for x in m1])
        l2_pvals = np.array([x['pval'] for x in m2])
        factor_names = np.array([(x['id'] + ': ' if show_factor_ids else '') + x['name'] for x in m1])

        return factor_names, l1_pvals, l2_pvals


    def plot_compare_module_enrichments(self, module_1, module_2, factor_type = 'motifs', hue = None, palette = 'coolwarm', hue_order = None, 
        ax = None, figsize = (7,7), legend_label = '', show_legend = True, fontsize = 12, pval_threshold = (1e-5, 1e-5),
        interactive = False, color = 'grey', label_closeness = 5, max_label_repeats = 5, show_factor_ids = False):

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = figsize)

        m1 = self.get_enrichments(module_1, factor_type)
        m2 = self.get_enrichments(module_2, factor_type)

        factor_names, l1_pvals, l2_pvals = self.join_factor_meta(m1, m2, show_factor_ids = show_factor_ids)
        
        plot_factor_influence(ax, np.array(l1_pvals)+1e-300, np.array(l2_pvals)+1e-300, factor_names, pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, label_closeness = label_closeness, 
            max_label_repeats = max_label_repeats, 
            axlabels = ('Module {} Enrichments'.format(str(module_1)),'Module {} Enrichments'.format(str(module_2))), 
            fontsize = fontsize, interactive = False, color = color)