
import torch
import logging
import tqdm
import pyro
from pyro.nn.module import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide.initialization import init_to_mean
from lisa import FromRegions
from lisa.core import genome_tools
from lisa.core.data_interface import DataInterface
from collections import Counter
import numpy as np
import pickle
from scipy import sparse
import re
from kladi.matrix_models.scipm_base import EarlyStopping
import glob

logger = logging.getLogger(__name__)

class LogMock:

    def __init__(self):
        pass

    def append(self, *args, **kwargs):
        logger.debug(*args)

class BadGeneException(Exception):
    pass


class RPModeler(FromRegions):

    rp_decay = 60000

    @classmethod
    def save_models(cls, models, prefix):
        for model in models:
            model.save(prefix + model.name + '.rpmodel')

    @classmethod
    def load_models(cls, prefix):
        models = []
        for model in glob.glob(prefix + '*.rpmodel'):
                models.append(RPModelPointEstimator.from_file(model))

        return models

    def __init__(self, species, accessibility_model, expression_model):

        assert(species in ['mm10','hg38'])
        regions = accessibility_model.peaks.tolist()
        assert(isinstance(regions, (list, tuple))), '"regions" parameter must be list of region tuples in format [ (chr,start,end [,score]), (chr,start,end [,score]) ... ] or name of bed file.'
        
        self.accessibility_model = accessibility_model
        self.expression_model = expression_model

        self.log = LogMock()
        self.data_interface = DataInterface(species, window_size=100, make_new=False,
            download_if_not_exists=False, log=self.log, load_genes=True,
            path = '.emptyh5.h5')

        self.num_regions_supplied = len(regions)

        regions = self._check_region_specification(regions)

        self.region_set = genome_tools.RegionSet(regions, self.data_interface.genome)
        self.region_score_map = np.array([r.annotation for r in self.region_set.regions])

        self.rp_map = 'basic'
        self.rp_map, self.genes, self.peaks = self.get_rp_map()

    def get_allowed_genes(self):

        return [x[3] for x in self.genes]

    def _check_region_specification(self, regions):

        invalid_chroms = Counter()
        valid_regions = []
        for i, region in enumerate(regions):
            assert(isinstance(region, (tuple, list)) and len(region) == 3), 'Error at region #{}: Each region passed must be in format (string \"chr\",int start, int end'\
                .format(str(i))
            
            try:
                new_region = genome_tools.Region(*region, annotation = i)
                self.data_interface.genome.check_region(new_region)
                valid_regions.append(new_region)
            except ValueError:
                raise AssertionError('Error at region #{}: Could not coerce positions into integers'.format(str(i)))
            except genome_tools.NotInGenomeError as err:
                invalid_chroms[region[0]]+=1
                #raise AssertionError('Error at region #{}: '.format(str(i+1)) + str(err) + '\nOnly main chromosomes (chr[1-22,X,Y] for hg38, and chr[1-19,X,Y] for mm10) are permissible for LISA test.')
            except genome_tools.BadRegionError as err:
                raise AssertionError('Error at region #{}: '.format(str(i+1)) + str(err))
        
        if len(invalid_chroms) > 0 :
            self.log.append('WARNING: {} regions encounted from unknown chromsomes: {}'.format(
                str(sum(invalid_chroms.values())), str(','.join(invalid_chroms.keys()))
            ))

        return valid_regions

    def _get_gene_model(self, gene_symbol, naive = False):

        try:
            gene_idx = np.argwhere(np.array([x[3] for x in self.genes]) == gene_symbol)[0]
        except IndexError:
            raise BadGeneException('Gene {} not in RefSeq database for this species'.format(gene_symbol))

        region_mask = self.rp_map[gene_idx, :].tocsr().indices

        try:
            region_distances = -self.rp_decay * np.log2(np.array(self.rp_map[gene_idx, region_mask])).reshape(-1)
        except TypeError:
            raise BadGeneException('No adjacent peaks to gene {}'.format(gene_symbol))

        tss_start = self.genes[gene_idx[0]][1]
        strand = self.genes[gene_idx[0]][4]

        upstream_mask = []
        for peak_idx in region_mask:

            peak = self.peaks[peak_idx]
            start, end = peak[1], peak[2]

            if strand == '+':
                upstream_mask.append(start <= tss_start)
            else:
                upstream_mask.append(end >= tss_start)

        promoter_mask = region_distances <= 1500
        upstream_mask = np.logical_and(upstream_mask, ~promoter_mask)
        downstream_mask  = np.logical_and(~upstream_mask, ~promoter_mask)

        region_distances = np.where(np.logical_or(upstream_mask, downstream_mask), region_distances - 1500, region_distances)

        if not naive:
            return PyroRPVI(gene_symbol, 
                region_distances = region_distances,
                upstream_mask = upstream_mask,
                downstream_mask = downstream_mask,
                region_score_map = self.region_score_map,
                region_mask = region_mask
            )
        else:
            return RPModelPointEstimator(gene_symbol,
                a_up = 1., a_down=1., a_promoter=1.,
                distance_up=np.e, distance_down=np.e, b = 0,
                upstream_mask=upstream_mask,downstream_mask=downstream_mask,
                promoter_mask=~np.logical_or(upstream_mask, downstream_mask),
                upstream_distances=region_distances[upstream_mask],
                downstream_distances=region_distances[downstream_mask],
                region_mask = region_mask,
                region_score_map=self.region_score_map
            )

    def _get_latent_vars(self, model, latent_comp = None, matrix = None):

        assert(not latent_comp is None or not matrix is None)

        if latent_comp is None:
            latent_compositions = model.predict(matrix)
        else:
            latent_compositions = latent_comp
            model._check_latent_vars(latent_compositions)

        return latent_compositions.copy()

    def add_accessibility_params(self, gene_models, accessibility = None, latent_compositions = None):
        logger.debug('Adding peak accessibility to models ...')

        latent_compositions = self._get_latent_vars(self.accessibility_model, matrix = accessibility,
            latent_comp= latent_compositions)

        for imputed_peaks in self.accessibility_model._batch_impute(latent_compositions):
            for model in gene_models:
                model.add_accessibility_params(imputed_peaks)

    def add_expression_params(self, gene_models, raw_expression):
        logger.debug('Adding modeled expression data to models ...')
        raw_expression = self.expression_model._validate_data(raw_expression)
        read_depth = self.expression_model._get_expression_distribution_parameters(raw_expression)

        for model in gene_models:
            try:
                gene_idx = np.argwhere(self.expression_model.genes == model.name)[0]
            except IndexError:
                raise Exception('Gene {} not in RefSeq database for this species'.format(model.name))
            model.add_expression_params(raw_expression[:,gene_idx], read_depth)

    def get_naive_models(self, gene_symbols):

        gene_models = []
        for symbol in gene_symbols:
            try:
                gene_models.append(self._get_gene_model(symbol, naive=True))
            except BadGeneException as err:
                logger.warn(str(err))

        return gene_models

    def train(self, gene_symbols, raw_expression, accessibility_matrix = None, 
        accessibility_latent_compositions = None, iters = 150, learning_rate = 0.1):

        gene_models = []
        for symbol in gene_symbols:
            try:
                gene_models.append(self._get_gene_model(symbol))
            except BadGeneException as err:
                logger.warn(str(err))

        self.add_accessibility_params(gene_models, accessibility = accessibility_matrix, 
            latent_compositions = accessibility_latent_compositions)
        self.add_expression_params(gene_models, raw_expression)

        logger.debug('Training RP models ...')
        trained_models = [model.train(max_iters = iters, learning_rate = learning_rate) for model in tqdm.tqdm(gene_models, desc = 'Training models')]

        return trained_models

    def predict(self, gene_models, accessibility_matrix = None, accessibility_latent_compositions = None):

        latent_compositions = self._get_latent_vars(self.accessibility_model, accessibility_latent_compositions, accessibility_matrix)

        batch_predictions = []
        for imputed_peaks in self.accessibility_model._batch_impute(latent_compositions):
            batch_predictions.append(
                np.hstack([
                    model.predict(imputed_peaks) for model in gene_models
                ])
            )

        return np.vstack(batch_predictions)     

class PyroRPVI(PyroModule):
    
    def __init__(self, gene_name, *, region_distances, upstream_mask, downstream_mask, 
        region_score_map, region_mask, use_cuda = False):
        super().__init__()

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logger.debug('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.to(self.device)

        self.region_score_map = region_score_map
        self.region_mask = region_mask
        self.name = gene_name
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = ~np.logical_or(self.upstream_mask, self.downstream_mask)

        self.upstream_distances = self.to_tensor(region_distances[np.newaxis, self.upstream_mask])
        self.downstream_distances = self.to_tensor(region_distances[np.newaxis, self.downstream_mask])
        
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)

        self.upstream_weights = []
        self.downstream_weights = []
        self.promoter_weights = []


    def add_accessibility_params(self, region_weights):

        region_weights = region_weights[:, self.region_score_map[self.region_mask]] * 1e4

        self.upstream_weights.append(region_weights[:, self.upstream_mask])
        self.downstream_weights.append(region_weights[:, self.downstream_mask])
        self.promoter_weights.append(region_weights[:, self.promoter_mask])

    def add_expression_params(self, raw_expression, read_depth):

        self.gene_expr = self.to_tensor(raw_expression)
        self.read_depth = self.to_tensor(read_depth.reshape(-1))

    def to_tensor(self, x, requires_grad = False):
        return torch.tensor(x, requires_grad = False).to(self.device)

    def RP(self, weights, distances, d):
        #print(weights, distances, d)
        return (weights * torch.pow(0.5, distances/(1e3 * d))).sum(-1)
    
    def forward(self, idx):

        with pyro.plate(self.name +"_regions", 3):
            a = pyro.sample(self.name +"_a", dist.HalfNormal(12.))

        with pyro.plate(self.name +"_upstream-downstream", 2):
            d = torch.exp(pyro.sample(self.name +'_logdistance', dist.Normal(np.e, 10.)))

        b = pyro.sample(self.name +"_b", dist.Normal(-10.,3.))
        theta = pyro.sample(self.name +"_theta", dist.Gamma(2., 0.5))

        with pyro.plate(self.name +"_data", len(idx), subsample_size=64) as ind:
            
            ind = torch.tensor(idx[ind])
            expr_rate = a[0] * self.RP(self.upstream_weights.index_select(0, ind), self.upstream_distances, d[0])\
                + a[1] * self.RP(self.downstream_weights.index_select(0, ind), self.downstream_distances, d[1]) \
                + a[2] * self.promoter_weights.index_select(0, ind).sum(-1) \
                + b
            
            mu = torch.multiply(self.read_depth.index_select(0, ind), torch.exp(expr_rate))
            p = torch.minimum(mu / (mu + theta), torch.tensor([0.99999]))
           
            pyro.sample(self.name + '_obs', dist.NegativeBinomial(total_count = theta, probs = p), obs = self.gene_expr.index_select(0, ind).reshape(-1))

            '''pyro.sample(self.name +'_obs', 
                        dist.ZeroInflatedNegativeBinomial(total_count=theta, probs=p, gate_logits = self.dropout_rate.index_select(0, ind).reshape(-1)),
                        obs= self.gene_expr.index_select(0, ind).reshape(-1))'''


    def train(self, learning_rate = 0.1, max_iters = 200, decay = 100, test_size = 0.2):

        logger.debug('Training {} RP model ...'.format(str(self.name)))
        pyro.clear_param_store()

        if isinstance(self.upstream_weights, list):
            #make into tensors
            self.upstream_weights = self.to_tensor(np.vstack(self.upstream_weights))
            self.downstream_weights = self.to_tensor(np.vstack(self.downstream_weights))
            self.promoter_weights = self.to_tensor(np.vstack(self.promoter_weights))

        self.N = len(self.read_depth)
        is_test_set = np.random.rand(self.N) < test_size
        self.test_idx = np.argwhere(is_test_set)[:,0]
        self.train_idx = np.argwhere(~is_test_set)[:,0]

        self.guide = AutoDiagonalNormal(self, init_loc_fn = init_to_mean)
        
        #adam = pyro.optim.Adam({"lr": learning_rate})
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': learning_rate}, 'gamma': 0.95})

        svi = SVI(self, self.guide, scheduler, loss=Trace_ELBO())

        early_stopper = EarlyStopping(patience = 3, tolerance = 1e-4)

        self.training_loss, self.testing_loss = [],[]
        for j in range(int(max_iters)):

            self.training_loss.append(
                float(svi.step(self.train_idx)/len(self.train_idx))
            )

            self.testing_loss.append(
                float(svi.evaluate_loss(self.test_idx)/len(self.test_idx))
            )

            if early_stopper.should_stop_training(self.testing_loss[-1]):
                logger.debug('Stopped training early!')
                break
    
        return self.get_MAP_estimator()

    def get_MAP_estimator(self):
        
        params = list(self.guide.get_posterior().mean.detach().cpu().numpy())
        a = params[:3]
        distance = params[3:5]
        b = params[5]

        return RPModelPointEstimator(
            self.name,
            a_up = a[0], a_down = a[1], a_promoter = a[2],
            distance_up = distance[0], distance_down = distance[1],
            upstream_mask = self.upstream_mask,
            downstream_mask = self.downstream_mask,
            promoter_mask  = self.promoter_mask,
            region_mask = self.region_mask, 
            region_score_map = self.region_score_map,
            upstream_distances = self.to_numpy(self.upstream_distances),
            downstream_distances = self.to_numpy(self.downstream_distances),
            b = b
        )
    
    @staticmethod
    def to_numpy(X):
        return X.detach().cpu().numpy()


class RPModelPointEstimator:

    @classmethod
    def from_file(cls, filename):

        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)

        gene = state_dict['name']
        del state_dict['name']

        return cls(gene, **state_dict)

    def __init__(self, gene_symbol, *, a_up, a_down, a_promoter, distance_up, distance_down, b,
            upstream_mask, downstream_mask, promoter_mask, upstream_distances, downstream_distances,
            region_mask, region_score_map):

        self.name = gene_symbol
        self.a_up = a_up
        self.a_down = a_down
        self.a_promoter = a_promoter
        self.distance_up = distance_up
        self.distance_down = distance_down
        self.b = b

        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = promoter_mask
        self.upstream_distances = upstream_distances
        self.downstream_distances = downstream_distances
        self.region_mask = region_mask
        self.region_score_map = region_score_map


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def predict(self, region_weights):

        region_weights = region_weights[:, self.region_score_map[self.region_mask]] * 1e4

        return self._get_log_expr_rate(
            upstream_weights=region_weights[:, self.upstream_mask], downstream_weights=region_weights[:, self.downstream_mask],
            promoter_weights=region_weights[:, self.promoter_mask]
        )[:, np.newaxis]

    def _get_log_expr_rate(self, *, upstream_weights, downstream_weights, promoter_weights, return_components = False):

        upstream_effects =  self.a_up * np.sum(upstream_weights * np.power(0.5, self.upstream_distances/ (1e3 * np.exp(self.distance_up)) ), axis = 1)
        downstream_effects = self.a_down * np.sum(downstream_weights * np.power(0.5, self.downstream_distances/ (1e3 * np.exp(self.distance_down))), axis = 1)
        promoter_effects = self.a_promoter * np.sum(promoter_weights, axis = 1)
        
        lam = upstream_effects + promoter_effects + downstream_effects + self.b

        if return_components:
            return lam, upstream_effects, promoter_effects, downstream_effects
        else:
            return lam


    def posterior_ISD(self, reg_state, motif_hits, qvalue = False):

        #assert(motif_hits.shape[1] == reg_state.shape[0])
        assert(len(reg_state.shape) == 1)

        reg_state = reg_state[self.region_mask][np.newaxis, :]

        if qvalue:
            motif_hits = motif_hits[:,self.region_mask]
            motif_hits.data = 1/np.exp(motif_hits.data)
            isd_mask = 1 - np.array(motif_hits.todense())
        else:
            motif_hits = np.array(motif_hits[:, self.region_mask].todense())
            isd_mask = np.maximum(1 - motif_hits, 0)

        isd_states = np.vstack((reg_state, reg_state * isd_mask))

        rp_scores = np.exp(self.get_log_expr_rate(isd_states))

        return 1 - rp_scores[1:]/rp_scores[0]

    def __str__(self):

        return '\n'.join(['{}: {}'.format(str(k), str(self.__dict__[k])) 
                for k in ['distance_up','distance_down','a_up','a_promoter','a_down']
        ])