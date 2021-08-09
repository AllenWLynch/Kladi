
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
from pyro.nn import PyroParam
import pyro.distributions as dist
from kladiv2.topic_model.base import BaseModel, get_fc_stack
from pyro.contrib.autoname import scope
import pyro
import numpy as np
import warnings
from scipy.sparse import isspmatrix
import json
import requests
from itertools import zip_longest
import configparser
from functools import partial

config = configparser.ConfigParser()
config.read('kladi/matrix_models/config.ini')

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

class ExpressionEncoder(nn.Module):

    def __init__(self,*,num_endog_features, num_topics, hidden, dropout, num_layers):
        super().__init__()
        output_batchnorm_size = 2*num_topics + 2

        self.num_topics = num_topics
        self.fc_layers = get_fc_stack(
            layer_dims = [num_endog_features + 1, *[hidden]*(num_layers-1), output_batchnorm_size],
            dropout = dropout, skip_nonlin = True
        )
        
    def forward(self, X):

        X = self.fc_layers(X)

        theta_loc = X[:, :self.num_topics]
        theta_scale = F.softplus(X[:, self.num_topics:(2*self.num_topics)])# + 1e-5

        rd_loc = X[:,-2].reshape((-1,1))
        rd_scale = F.softplus(X[:,-1]).reshape((-1,1))# + 1e-5

        return theta_loc, theta_scale, rd_loc, rd_scale


def ExpressionModel(BaseModel):

    encoder_model = ExpressionEncoder

    @property
    def genes(self):
        return self.features

    @staticmethod
    def _residual_transform(y_ij, pi_j_hat):
        
        n_i = y_ij.sum(axis = 1, keepdims = True)

        mu_ij_hat = n_i * pi_j_hat[np.newaxis, :]

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


    def _get_weights(self):
        super()._get_weights()

        self.dispersion = PyroParam(torch.ones(self.exog_features) * 5, constraint = constraints.positive)


    @scope(prefix= 'rna')
    def model(self,*,endog_features, exog_features, read_depth):
        theta_loc, theta_scale = super().model(endog_features = endog_features, exog_features = exog_features)
        pyro.module("decoder", self.decoder)

        with pyro.plate("cells", endog_features.shape[0]):

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)
            
            expr_rate = self.decoder(theta)

            read_scale = pyro.sample('read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1))
            
            mu = torch.multiply(read_scale, expr_rate)
            p = mu / (mu + self.dispersion)

            pyro.sample('obs', dist.NegativeBinomial(total_count = self.dispersion, probs = p).to_event(1), obs = exog_features)


    @scope(prefix= 'rna')
    def guide(self, endog_features, exog_features, read_depth):
        super().guide(endog_features = endog_features, exog_features = exog_features, read_depth = read_depth)
        pyro.module("encoder", self.encoder)

        endog_features = torch.cat([endog_features, torch.log(read_depth)])

        with pyro.plate("cells", endog_features.shape[0]):
            
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(endog_features)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1)
            )

            read_depth = pyro.sample(
                "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1)
            )

    def _preprocess_endog(self, X):
        
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if isspmatrix(X):
            X = X.toarray()

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_endog_features)

        try:
            self.residual_pi
        except AttributeError:
            self.residual_pi = X.sum(axis = 0)/X.sum()
        
        assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        X = self._residual_transformation(X.astype(np.float32), self.residual_pi)

        return torch.tensor(X, requires_grad = False).to(self.device)


    def _preprocess_exog(self, X):
        
        assert(isinstance(X, np.ndarray) or isspmatrix(X))
        
        if isspmatrix(X):
            X = X.toarray()

        assert(len(X.shape) == 2)
        assert(X.shape[1] == self.num_exog_features)
        
        assert(np.isclose(X.astype(np.int64), X, 1e-2).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'
        return torch.tensor(X.astype(np.float32), requires_grad = False).to(self.device)
        
    
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
            gene_scores = self._score_features()[module_num,:]
            top_genes_mask = gene_scores - gene_scores.mean() > 2

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
    

    def _post_genelist(self, module_num, top_n = None):
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

        response = requests.post(enrichr_url + post_endpoint, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        list_id = json.loads(response.text)['userListId']
        return list_id

    def _get_ontology(self, list_id, ontology = 'WikiPathways_2019_Human'):
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


    def _get_enrichments(self, list_id, ontologies = config.get('Enrichr','ontologies').split(',')):
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


    def _plot_enrichments(self, enrichment_results, show_genes = True, show_top = 5, barcolor = 'lightgrey',
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