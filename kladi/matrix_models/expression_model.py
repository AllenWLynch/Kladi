
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.constraints as constraints
from tqdm import tqdm
import warnings

from scipy.sparse import isspmatrix
from kladi.matrix_models.scipm_base import BaseModel
import configparser
import requests
import json
from itertools import zip_longest
import matplotlib.pyplot as plt
import logging

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

class GeneDevianceModel:

    def __init__(self, highly_variable):
        self.highly_variable = highly_variable

    def fit(self, y_ij):
        
        y_ij = y_ij[:, self.highly_variable]

        logging.info('Learning deviance featurization for transcript counts ...')
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

        return np.nan_to_num(r_ij)


class ExpressionEncoder(nn.Module):
    # Base class for the encoder net, used in the guide
    # Base class for the encoder net, used in the guide
    def __init__(self, num_genes, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(num_genes + 1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.fcrd = nn.Linear(hidden, 2)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnrd = nn.BatchNorm1d(2)

    def forward(self, inputs):
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = F.softplus(theta_scale) #(0.5 * theta_scale).exp()  # Enforces positivity
        
        rd = self.bnrd(self.fcrd(h))
        rd_loc = rd[:,0]
        rd_scale = F.softplus(rd[:,1]) #(0.5 * rd[:,1]).exp()

        return theta_loc, theta_scale, rd_loc, rd_scale

class ExpressionDecoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, num_genes, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_genes, bias = False)
        self.bn = nn.BatchNorm1d(num_genes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_topics, num_genes, bias = False)
        self.bn2 = nn.BatchNorm1d(num_genes)

    def forward(self, latent_composition):
        inputs = self.drop(latent_composition)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1), self.bn2(self.fc(inputs))


class ExpressionModel(BaseModel):

    @classmethod
    def param_search(cls,*,counts, genes, num_modules = [5, 10, 15, 20, 25], initial_counts = 10, dropout = 0.2, hidden = 128, use_cuda = True,
        num_epochs = 100, batch_size = 32, learning_rate = 1e-3, eval_every = 1, test_proportion = 0.05 
    ):
        
        assert(isinstance(num_modules, (list, np.ndarray)))
        testing_loss = []
        for K in num_modules:
            assert(isinstance(K, int) and K > 0)
            logging.info('Training model with {} modules ...'.format(str(K)))
            logging.info('----------------------------------')
            test_model = cls(genes, num_modules=K, initial_counts = initial_counts, dropout=dropout, hidden=hidden, use_cuda=use_cuda)
            test_model.fit(counts, num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate, eval_every = eval_every, 
                test_proportion = test_proportion, use_validation_set = True)

            testing_loss.append(test_model.testing_loss[-1])

        return testing_loss

    def __init__(self, genes, highly_variable = None, num_modules = 15, initial_counts = 10, 
        dropout = 0.2, hidden = 128, use_cuda = True):

        assert(isinstance(genes, (list, np.ndarray)))
        self.genes = np.ravel(np.array(genes))
        self.num_genes = len(self.genes)

        if highly_variable is None:
            highly_variable = np.ones_like(genes).astype(bool)
        else:
            assert(isinstance(highly_variable, np.ndarray))
            assert(highly_variable.dtype == bool)
            
            highly_variable = np.ravel(highly_variable)
            assert(len(highly_variable) == self.num_genes)
        self.highly_variable = highly_variable

        self.num_features = int(highly_variable.sum())
        
        super().__init__(self.num_features, 
            ExpressionEncoder(self.num_features, num_modules, hidden, dropout), 
            ExpressionDecoder(self.num_genes, num_modules, dropout), 
            num_topics = num_modules, initial_counts = initial_counts, 
            hidden = hidden, dropout = dropout, use_cuda = use_cuda)

        
    def model(self, raw_expr, encoded_expr, read_depth):

        pyro.module("decoder", self.decoder)

        dispersion = pyro.param("dispersion", torch.tensor(5.).to(self.device) * torch.ones(self.num_genes).to(self.device), 
            constraint = constraints.positive)
        
        with pyro.plate("cells", encoded_expr.shape[0]):

            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution
            theta_loc = self.prior_mu * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta_scale = self.prior_std * encoded_expr.new_ones((encoded_expr.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta/theta.sum(-1, keepdim = True)

            read_scale = pyro.sample(
                'read_depth', dist.LogNormal(torch.log(read_depth), 1.).to_event(1)
            )

            #read_scale = torch.minimum(read_scale, self.max_scale)
            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            expr_rate, dropout = self.decoder(theta)

            mu = torch.multiply(read_scale, expr_rate)
            p = torch.minimum(mu / (mu + dispersion), self.max_prob)

            pyro.sample('obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=dispersion, probs=p, gate_logits=dropout).to_event(1),
                        obs= raw_expr)


    def guide(self, raw_expr, encoded_expr, read_depth):

        pyro.module("encoder", self.encoder)

        with pyro.plate("cells", encoded_expr.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

            read_depth = pyro.sample(
                "read_depth", dist.LogNormal(rd_loc.reshape((-1,1)), rd_scale.reshape((-1,1))).to_event(1))

    def _get_expression_distribution_parameters(self, raw_expr, batch_size = 32):

        X = self._validate_data(raw_expr)
        assert(isinstance(batch_size, int) and batch_size > 0)

        read_depths, dropouts = [],[]
        for i,batch in enumerate(self._get_batches(X, batch_size = batch_size)):
            raw_expr, encoded_expr, read_depth = batch
            theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)
            latent_comp = theta_loc.exp()/theta_loc.exp().sum(-1, keepdim = True)

            expr, dropout = self.decoder(latent_comp)

            dropouts.append(dropout.detach().cpu().numpy())
            read_depths.append(np.exp(rd_loc.detach().cpu().numpy()))

        read_depth = np.concatenate(read_depths, 0)
        dropout = np.vstack(dropouts)

        return read_depth, dropout


    def _get_latent_MAP(self, raw_expr, encoded_expr, read_depth):
        theta_loc, theta_scale, rd_loc, rd_scale = self.encoder(encoded_expr)

        Z = theta_loc.cpu().detach().numpy()
        return np.exp(Z)/np.exp(Z).sum(-1, keepdims = True)


    def _get_batches(self, count_matrix, batch_size = 32, bar = False, training = True):
        
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
        assert(X.shape[1] == self.num_genes)
        
        assert(np.isclose(X.astype(np.int64), X).all()), 'Input data must be raw transcript counts, represented as integers. Provided data contains non-integer values.'

        return X.astype(np.float32)

    def impute(self, latent_compositions):

        assert(isinstance(latent_compositions, np.ndarray))
        assert(len(latent_compositions.shape) == 2)
        assert(latent_compositions.shape[1] == self.num_topics)
        assert(np.isclose(latent_compositions.sum(-1), 1).all())

        latent_compositions = self._to_tensor(latent_compositions)

        return self.decoder(latent_compositions)[0].cpu().detach().numpy()

    def save(self, filename):
        
        state_dict = dict(state = self.state_dict())
        try:
            state_dict['deviance_pi'] = self.deviance_model.pi_j_hat
        except AttributeError:
            raise Exception('Cannot save model before it is trained.')

        state_dict['genes'] = self.genes
        torch.save(state_dict, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state'])
        self.eval()       

        self.deviance_model = GeneDevianceModel(self.highly_variable)
        self.deviance_model.set_pi(state['deviance_pi'])
        self.set_device('cpu')
        #self.genes = state['genes']


    def _featurize(self, count_matrix):

        encoded_counts = self.deviance_model.transform(count_matrix)
        read_depth = count_matrix.sum(-1, keepdims = True)

        encoded_counts = np.hstack([encoded_counts, np.log(read_depth)])

        return self._to_tensor(count_matrix), self._to_tensor(encoded_counts), self._to_tensor(read_depth)

    def rank_genes(self, module_num):
        assert(isinstance(module_num, int) and module_num < self.num_topics and module_num >= 0)

        return self.genes[np.argsort(self._get_beta()[module_num, :])]

    def get_top_genes(self, module_num, top_n = 200):
        return self.rank_genes(module_num)[-top_n : ]


    def rank_modules(self, gene):
        
        assert(gene in self.genes)

        gene_idx = np.argwhere(self.genes == gene)[0]
        return list(sorted(zip(range(self.num_topics), self._get_beta()[:, gene_idx]), key = lambda x : x[1]))
    
    def _get_beta(self):
        return super()._get_beta()[:, :-1]

    def post_genelist(self, module_num, top_n_genes = 200):

        top_genes = '\n'.join(self.get_top_genes(module_num, top_n=top_n_genes))

        enrichr_url = config.get('Enrichr','url')
        post_endpoint = config.get('Enrichr','post')

        payload = {
            'list': (None, top_genes),
        }

        logging.info('Querying Enrichr with module {} genes.'.format(str(module_num)))
        response = requests.post(enrichr_url + post_endpoint, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        list_id = json.loads(response.text)['userListId']
        return list_id

    def get_ongology(self, list_id, ontology = 'WikiPathways_2019_Human'):

        enrichr_url = config.get('Enrichr','url')
        get_endpoint = config.get('Enrichr','get').format(list_id = list_id, ontology = ontology)

        response = requests.get(enrichr_url + get_endpoint)
        if not response.ok:
            raise Exception('Error fetching enrichment results')
        
        data = json.loads(response.text)[ontology]

        headers = config.get('Enrichr','results_headers').split(',')
        
        return [dict(zip(headers, x)) for x in data]


    def get_enrichments(self, list_id):

        logging.info('Downloading results ...')

        enrichments = {
            ontology : self.get_ongology(list_id, ontology=ontology)
            for ontology in config.get('Enrichr','ontologies').split(',')   
        }

        return enrichments

    def plot_enrichments(self, enrichment_results, show_genes = True, figsize = (20,20), show_top = 5, barcolor = 'lightgrey',
        text_color = 'black', return_fig = False):
        
        num_plots = len(enrichment_results.keys())

        fig, ax = plt.subplots(num_plots//2, 2, figsize = figsize)
        ax = np.ravel(ax)
        for i, (key, results) in enumerate(enrichment_results.items()):
        
            terms, genes, pvals = [],[],[]
            for result in results[:show_top]:
                
                terms.append(
                    compact_string(result['term'])
                )        
                genes.append(' '.join(result['genes']))
                pvals.append(-np.log10(result['pvalue']))
                
            g = ax[i].barh(np.arange(len(terms)), pvals, color=barcolor)
            ax[i].set_yticks(np.arange(len(terms)))
            ax[i].set_yticklabels(terms)
            ax[i].invert_yaxis()
            ax[i].set(title = key, xlabel = '-log10 pvalue')
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].yaxis.set_ticks_position('left')
            ax[i].xaxis.set_ticks_position('bottom')
            
            if show_genes:
                for j, p in enumerate(ax[i].patches):
                    _x = p.get_x() + p.get_width()
                    _y = p.get_y() + p.get_height() - p.get_height()/3
                    ax[i].text(0.1, _y, compact_string(genes[j], max_wordlen=10, join_spacer = ', '), ha="left", color = text_color)

        plt.tight_layout()
        if return_fig:
            return fig, ax