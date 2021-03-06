
from os import stat
import numpy as np
import pyro
from pyro import poutine
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints
from tqdm import tqdm
import warnings

from scipy.sparse import isspmatrix
from kladi.matrix_models.scipm_base import BaseModel, get_fc_stack
import configparser
import requests
import json
from itertools import zip_longest
import matplotlib.pyplot as plt
import logging
from math import ceil
from kladi.core.plot_utils import map_plot
from functools import partial
from kladi.matrix_models.scipm_base import Decoder
from pyro.contrib.autoname import scope

config = configparser.ConfigParser()
config.read('kladi/matrix_models/config.ini')

logger = logging.getLogger(__name__)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def compact_string(x, max_wordlen = 4, join_spacer = ' ', sep = ' '):
    return '\n'.join(
        [
            join_spacer.join([x for x in segment if not x == '']) for segment in grouper(x.split(sep), max_wordlen, fillvalue='')
        ]
    )

class GeneDevianceModel:

    def __init__(self, highly_variable):
        self.highly_variable = highly_variable

    def fit(self, y_ij):
        
        y_ij = y_ij[:, self.highly_variable]
        self.pi_j_hat = y_ij.sum(axis = 0)/y_ij.sum()

        return self

    def set_pi(self, pi):
        self.pi_j_hat = pi

    def transform(self, y_ij):
        
        y_ij = y_ij[:, self.highly_variable]
        
        n_i = y_ij.sum(axis = 1, keepdims = True)

        mu_ij_hat = n_i * self.pi_j_hat[np.newaxis, :]

        count_dif = n_i - y_ij
        expected_count_dif = n_i - mu_ij_hat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            r_ij = np.multiply(
                np.sign(y_ij - mu_ij_hat), 
                np.sqrt(
                np.where(y_ij > 0, 2 * np.multiply(y_ij, np.log(y_ij / mu_ij_hat)), 0) + \
                2 * np.multiply(count_dif, np.log(count_dif / expected_count_dif))
                )
            )

        return np.clip(np.nan_to_num(r_ij), -10, 10)


class ExpressionEncoder(nn.Module):

    def __init__(self, num_genes, num_topics, hidden, dropout, num_layers):
        super().__init__()
        output_batchnorm_size = 2*num_topics + 2

        self.num_topics = num_topics
        self.fc_layers = get_fc_stack(
            layer_dims = [num_genes + 1, *[hidden]*(num_layers-1), output_batchnorm_size],
            dropout = dropout, skip_nonlin = True
        )
        
    def forward(self, X):

        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])# + 1e-5

        rd_loc = X[:,-2].reshape((-1,1))
        rd_scale = F.softplus(X[:,-1]).reshape((-1,1))# + 1e-5

        return theta_loc, theta_scale, rd_loc, rd_scale


class ExpressionModel(BaseModel):
    '''
    Class
    '''
    
    def __init__(self, genes, highly_variable = None, num_modules = 15, decoder_dropout = 0.2, 
        encoder_dropout = 0.1, hidden = 128, use_cuda = True, num_layers = 3, seed = None):
        '''
        Initialize ExpressionModel instance. 

        Example:

            >> genes[:3]
            ['GATA3', 'WNT3', 'CDK8']

            >> highly_variable[:3]
            np.array([True, False, False], dtype = bool)

            >> expr_model = ExpressionModel(genes, highly_variable = highly_variable, num_modules = 10)


        Args:
            genes (list, np.ndarray): Gene names / column names for count matrix, length must match dimension 2 of count matrix
            highly_variable (np.ndarray): boolean mask of same length as ``genes``. Genes flagged with ``True`` will be used as features for encoder. All genes will be used as features for decoder.
                This allows one to impute many genes while only learning modules on highly variable genes, decreasing model complexity and training time.
            num_modules (int): number of gene modules to find
            initial_counts (int): sparsity parameter, related to pseudocounts of dirichlet prior. Increasing will lead to denser cell latent variables, decreasing will lead to more sparse latent variables.
            dropout (float between 0,1): dropout rate for model.
            hidden (int): number of nodes in encoder hidden layers.
            use_cuda (bool): use CUDA to accelerate training on GPU (if GPU is available).

        Returns:
            ExpressionModel
        '''

        assert(isinstance(genes, (list, np.ndarray)))
        self.genes = np.ravel(np.array(genes))
        
        kwargs = dict(
            num_modules = num_modules,
            num_exog_features = len(self.genes),
            highly_variable = highly_variable,
            hidden = hidden,
            num_layers = num_layers,
            decoder_dropout = decoder_dropout,
            encoder_dropout = encoder_dropout,
            use_cuda = use_cuda,
            seed = seed,
        )
        
        super().__init__(ExpressionEncoder, Decoder, **kwargs)

    @scope(prefix= 'rna')
    def model(self, raw_expr, encoded_expr, read_depth, anneal_factor = 1.):

        pyro.module("decoder", self.decoder)

        self.dispersion = pyro.param("dispersion", torch.tensor(5.) * torch.ones(self.num_exog_features), 
            constraint = constraints.positive).to(self.device)

        _alpha, _beta = self._get_gamma_parameters(self.I, self.num_topics)
        with pyro.plate("topics", self.num_topics):
            initial_counts = pyro.sample("a", dist.Gamma(self._to_tensor(_alpha), self._to_tensor(_beta)))

        theta_loc = self._get_prior_mu(initial_counts, self.K)
        theta_scale = self._get_prior_std(initial_counts, self.K)

        #print(theta_loc, theta_scale)
        
        with pyro.plate("cells", encoded_expr.shape[0]):

            # Dirichlet prior  ????(????|????) is replaced by a log-normal distribution
            with poutine.scale(None, anneal_factor):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

                read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))

            theta = theta/theta.sum(-1, keepdim = True)
            expr_rate = self.decoder(theta)

            mu = torch.multiply(read_scale, expr_rate)
            p = torch.minimum(mu / (mu + self.dispersion), self.max_prob)

            pyro.sample('obs', dist.NegativeBinomial(total_count = self.dispersion, probs = p).to_event(1), obs = raw_expr)

    @scope(prefix= 'rna')
    def guide(self, raw_expr, encoded_expr, read_depth, anneal_factor = 1.):

        pyro.module("encoder", self.encoder)

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(*self._get_gamma_moments(self.I, self.num_topics))
        counts_mu = pyro.param('counts_mu', _counts_mu * encoded_expr.new_ones((self.num_topics,))).to(self.device)
        counts_std = pyro.param('counts_std', np.sqrt(_counts_var) * encoded_expr.new_ones((self.num_topics,)), 
                constraint = constraints.positive).to(self.device)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(counts_mu[k], counts_std[k]))

        
        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  ????(????|????) is replaced by a log-normal distribution,
            # where ?? and ?? are the encoder network outputs
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

            with poutine.scale(None, anneal_factor):
                theta = pyro.sample(
                    "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
                )

                read_depth = pyro.sample(
                    "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
                )
            

    def _get_expression_distribution_parameters(self, raw_expr, batch_size = 32):
        
        def detach(x):
            return x.detach().cpu().numpy()

        X = self._validate_data(raw_expr)
        assert(isinstance(batch_size, int) and batch_size > 0)

        rd_locs, rd_scales, softmax_denoms = [], [], []
        for i,batch in enumerate(self._get_batches(X, batch_size = batch_size)):
            raw_expr, encoded_expr, read_depth = batch
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

            rd_locs.append(detach(rd_loc))
            rd_scales.append(detach(rd_scale))

            theta = theta_loc.exp()/theta_loc.exp().sum(-1, keepdim = True)
            softmax_denoms.append(
                detach(self.decoder.get_softmax_denom(theta))
            )

        rd_loc = np.concatenate(rd_locs, 0)
        rd_scale = np.concatenate(rd_scales, 0)
        softmax_denom = np.concatenate(softmax_denoms, 0)
        return rd_loc, rd_scale, softmax_denom


    def _get_latent_MAP(self, raw_expr, encoded_expr, read_depth):
        theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

        Z = theta_loc.cpu().detach().numpy()
        return np.exp(Z)/np.exp(Z).sum(-1, keepdims = True)


    def _get_batches(self, count_matrix, batch_size = 32, bar = False, training = True, desc = None):
        
        N = len(count_matrix)
        
        try:
            self.deviance_model
        except AttributeError:
            self.deviance_model = GeneDevianceModel(self.highly_variable).fit(count_matrix)

        for batch_start, batch_end in self._iterate_batch_idx(N, batch_size):
            yield self._featurize(count_matrix[batch_start : batch_end, :])

    def _validate_data(self, X):
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if isspmatrix(X):
            X = np.array(X.todense())

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_exog_features)
        
        assert(np.isclose(X.astype(np.int64), X, 1e-1).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        return X.astype(np.float32)

    def impute(self, latent_compositions):
        '''
        Compute imputed gene expression values using cells' latent variable representations.

        Args:
            latent_compositions (np.npdarray): Cells x num_modules array, each row must sum to 1

        Returns:
            (np.ndarray): imputed expression, Cells x num_genes matrix
        '''

        assert(isinstance(latent_compositions, np.ndarray))
        assert(len(latent_compositions.shape) == 2)
        assert(latent_compositions.shape[1] == self.num_topics)
        assert(np.isclose(latent_compositions.sum(-1), 1).all())

        latent_compositions = self._to_tensor(latent_compositions)

        return self.decoder(latent_compositions).cpu().detach().numpy()

    def _get_save_data(self):
        return dict(
            pi = self.deviance_model.pi_j_hat,
            **super()._get_save_data()
        )

    def _load_save_data(self, data):
        super()._load_save_data(data)

        self.deviance_model = GeneDevianceModel(self.highly_variable)
        self.deviance_model.set_pi(data['pi'])

        return self


    def _featurize(self, count_matrix):

        encoded_counts = self.deviance_model.transform(count_matrix)
        read_depth = count_matrix.sum(-1, keepdims = True)

        encoded_counts = np.hstack([encoded_counts, np.log(read_depth)])

        return self._to_tensor(count_matrix), self._to_tensor(encoded_counts), self._to_tensor(read_depth)

    def rank_genes(self, module_num):
        '''
        Ranks genes according to their activation in module ``module_num``. Sorted from most suppressed to most activated.

        Args:
            module_num (int): For which module to rank genes

        Returns:
            np.ndarray: sorted array of gene names in order from most suppressed to most activated given the specified module
        '''
        assert(isinstance(module_num, int) and module_num < self.num_topics and module_num >= 0)

        return self.genes[np.argsort(self._score_features()[module_num, :])]

    def get_top_genes(self, module_num, top_n = None):
        '''
        For a module, return the top n genes that are most activated.

        Args:
            module_num (int): For which module to return most activated genes
            top_n (int): number of genes to return

        Returns
            (np.ndarray): Names of top n genes, sorted from least to most activated
        '''

        if top_n is None:
            top_genes_mask = self._score_features()[module_num,:] > 2

            if top_genes_mask.sum() > 200:
                return self.genes[top_genes_mask]
            else:
                top_n = 200

        assert(isinstance(top_n, int) and top_n > 0)
        return self.rank_genes(module_num)[-top_n : ]


    def rank_modules(self, gene):
        '''
        For a gene, rank how much its expression is activated by each module

        Args:
            gene (str): name of gene
        
        Raises:
            AssertionError: if ``gene`` is not in self.genes
        
        Returns:
            (list): of format [(module_num, activation), ...]
        '''
        
        assert(gene in self.genes)

        gene_idx = np.argwhere(self.genes == gene)[0]
        return list(sorted(zip(range(self.num_topics), self._score_features()[:, gene_idx]), key = lambda x : x[1]))
    

    def post_genelist(self, module_num, top_n = None):
        '''
        Post genelist to Enrichr, recieve genelist ID for later retreival.

        Args:
            module_num (int): which module's top genes to post
            top_n_genes (int): number of genes to post

        Returns:
            enrichr_id (str): unique ID of genelist for retreival with ``get_enrichments`` or ``get_ontology``
        '''

        top_genes = '\n'.join(self.get_top_genes(module_num, top_n=top_n))

        enrichr_url = config.get('Enrichr','url')
        post_endpoint = config.get('Enrichr','post')

        payload = {
            'list': (None, top_genes),
        }

        logger.info('Querying Enrichr with module {} genes.'.format(str(module_num)))
        response = requests.post(enrichr_url + post_endpoint, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        list_id = json.loads(response.text)['userListId']
        return list_id

    def get_ontology(self, list_id, ontology = 'WikiPathways_2019_Human'):
        '''
        Fetches the gene-set enrichments for a genelist in a certain ontology from Enrichr

        Args:
            list_id (str): unique ID of genelist from ``post_genelist``
            ontology (str, default = Wikipathways_2019_Human): For which ontology to download results

        Returns:
            (dict): enrichments, with format:

                {
                    ontology: {
                        rank : [...],
                        term : [...],
                        pvalue : [...],
                        zscore : [...],
                        combined_score : [...],
                        genes : [...],
                        adj_pvalue : [...]
                    }

                }
        '''

        enrichr_url = config.get('Enrichr','url')
        get_endpoint = config.get('Enrichr','get').format(list_id = list_id, ontology = ontology)

        response = requests.get(enrichr_url + get_endpoint)
        if not response.ok:
            raise Exception('Error fetching enrichment results')
        
        data = json.loads(response.text)[ontology]

        headers = config.get('Enrichr','results_headers').split(',')
        
        return {ontology : [dict(zip(headers, x)) for x in data]}


    def get_enrichments(self, list_id, ontologies = config.get('Enrichr','ontologies').split(',')):
        '''
        Fetches the gene-set enrichments for a genelist from ontologies listed

        Args:
            list_id (str): unique ID of genelist from ``post_genelist``
            ontologies (list, default in kladi/matrix_models/config.ini): or which ontologies to download results

        Returns:
            (dict): enrichments, with format:

                {
                    ontology: {
                        rank : [...],
                        term : [...],
                        pvalue : [...],
                        zscore : [...],
                        combined_score : [...],
                        genes : [...],
                        adj_pvalue : [...]
                    }
                    ...
                }
        '''

        logger.info('Downloading results ...')

        enrichments = dict()
        for ontology in ontologies:
            enrichments.update(self.get_ontology(list_id, ontology=ontology))

        return enrichments

    @staticmethod
    def _enrichment_plot(ax, ontology, results,*,
        text_color, show_top, barcolor, show_genes, max_genes):

        terms, genes, pvals = [],[],[]
        for result in results[:show_top]:
            
            terms.append(
                compact_string(result['term'])
            )        
            genes.append(' '.join(result['genes'][:max_genes]))
            pvals.append(-np.log10(result['pvalue']))
            
        ax.barh(np.arange(len(terms)), pvals, color=barcolor)
        ax.set_yticks(np.arange(len(terms)))
        ax.set_yticklabels(terms)
        ax.invert_yaxis()
        ax.set(title = ontology, xlabel = '-log10 pvalue')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        
        if show_genes:
            for j, p in enumerate(ax.patches):
                _y = p.get_y() + p.get_height() - p.get_height()/3
                ax.text(0.1, _y, compact_string(genes[j], max_wordlen=10, join_spacer = ', '), ha="left", color = text_color)


    def plot_enrichments(self, enrichment_results, show_genes = True, show_top = 5, barcolor = 'lightgrey',
        text_color = 'black', return_fig = False, enrichments_per_row = 2, height = 4, aspect = 2.5, max_genes = 15):
        '''
        Make plot of geneset enrichments given results from ``get_ontology`` or ``get_enrichments``.

        Example:

            post_id = expr_model.post_genelist(0) #post top 250 module 0 genes
            enrichments = expr_model.get_enrichments(post_id)
            expr_model.plot_enrichments(enrichments)

        Args:
            enrichment_results (dict): output from ``get_ontology`` or ``get_enrichments``
            show_genes (bool): overlay gene names on top of bars
            show_top (int): plot top n enrichment results
            barcolor (color): color of barplot bars
            text_color (text_color): color of text on barplot bars
            return_fig (bool): return fig and axes objects
            enrichments_per_row (int): number of plots per row
            height (float): height of each plot
            aspect (float): multiplier for width of each plot, width = aspect * height
            max_genes (int): maximum number of genes to display on bar

        Returns (if return_fig is True):
            matplotlib.figure, matplotlib.axes.Axes

        '''
        
        func = partial(self._enrichment_plot, text_color = text_color, 
            show_top = show_top, barcolor = barcolor, show_genes = show_genes, max_genes = max_genes)

        fig, ax = map_plot(func, enrichment_results.keys(), enrichment_results.values(), plots_per_row = enrichments_per_row, 
            height =height, aspect = aspect)  
            
        plt.tight_layout()
        if return_fig:
            return fig, ax