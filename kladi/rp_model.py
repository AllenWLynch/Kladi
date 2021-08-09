
import torch
import logging
import tqdm
import pyro
from pyro import poutine
from pyro.nn.module import PyroModule, PyroSample, PyroParam
from pyro.infer.autoguide import AutoDelta
import pyro.distributions as dist
from torch.optim import SGD
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_mean, init_to_mean
from lisa import FromRegions
from lisa.core import genome_tools
from lisa.core.data_interface import DataInterface
from collections import Counter
import numpy as np
import pickle
from scipy import sparse, stats
import re
from kladi.matrix_models.scipm_base import EarlyStopping
import glob
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator
from functools import partial, wraps
import time
from pyro.contrib.autoname import scope
from scipy.stats import chi2, nbinom
from kladi.covISD import ProbISD_Results

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def init_to_value(
    site = None,
    values = {},
    *,
    fallback = init_to_mean,
):
    if site is None:
        return partial(init_to_value, values=values, fallback=fallback)

    if site["name"] in values:
        return values[site["name"]]
    if fallback is not None:
        return fallback(site)
    raise ValueError(f"No init strategy specified for site {repr(site['name'])}")

class LogMock:

    def __init__(self):
        pass

    def append(self, *args, **kwargs):
        logger.debug(*args)

class BadGeneException(Exception):
    pass

class ModelingError(Exception):
    pass

def _warmup_lr_decay(step,*,n_batches_per_epoch, lr_decay):
    return min(1., (step + 1)/(n_batches_per_epoch + 1))*(lr_decay**int(max(0, step - n_batches_per_epoch)))


def _get_latent_vars(model, latent_comp = None, matrix = None):

    assert(not latent_comp is None or not matrix is None)

    if latent_comp is None:
        latent_compositions = model.predict(matrix)
    else:
        latent_compositions = latent_comp
        model._check_latent_vars(latent_compositions)

    return latent_compositions.copy()

def get_expression_distribution(f_Z,*, batchnorm, rd_loc, rd_scale, softmax_denom, 
    dispersion, gamma, bias, deterministic = True):

    if not deterministic:
        read_depth = torch.distributions.LogNormal(rd_loc, rd_scale).sample()
    else:
        read_depth = rd_loc.exp()

    prediction = batchnorm(f_Z)
    
    independent_rate = (gamma * prediction + bias).exp()
    pyro.deterministic('independent_rate', independent_rate)

    rate =  independent_rate/softmax_denom

    mu = read_depth * rate

    p = torch.minimum(mu / (mu + dispersion), torch.tensor([0.99999], requires_grad = False))

    pyro.deterministic('prob_success', p)
    return dist.NegativeBinomial(total_count = dispersion, probs = p)


def register_input(func):

    @wraps(func)
    def process_input(self,*args, accessibility_latent_compositions=None, 
        accessibility_matrix=None, expression = None, idx = None, **kwargs):
        
        if accessibility_latent_compositions is None and accessibility_matrix is None \
                and expression is None:
                assert(all([model.has_features() for model in self.gene_models])), 'Must provide features before running an inference function.'
                return func(self, idx = idx, *args, **kwargs)
        else:


            assert(not expression is None)
            latent_comp = _get_latent_vars(self.accessibility_model, latent_comp = accessibility_latent_compositions,
                matrix = accessibility_matrix)

            self.add_accessibility_params(accessibility_matrix, latent_comp)
            self.add_expression_params(expression)

            list(map(lambda x : x._fix_features(), self.gene_models))

            return func(self, idx = idx, *args, **kwargs)
    
    return process_input


class CisModeler(FromRegions):

    rp_decay = 60000

    def __init__(self, species,*,accessibility_model, expression_model, genes = None,  model_type = 'cis',
        max_iters = 150, learning_rate = 0.1, use_SGD = False, test_size = 0.2, restarts = 5,
        batch_size = 64, decay = 0.95, momentum = 0.1, cis_models = None):

        assert(species in ['mm10','hg38'])
        regions = accessibility_model.peaks.tolist()
        assert(isinstance(regions, (list, tuple))), '"regions" parameter must be list of region tuples in format [ (chr,start,end [,score]), (chr,start,end [,score]) ... ] or name of bed file.'
        
        self.accessibility_model = accessibility_model
        self.expression_model = expression_model

        self.use_SGD = use_SGD
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.decay = decay
        self.momentum = momentum
        self.test_size = test_size
        self.restarts = restarts
        self.cis_models = cis_models
        self.model_type = model_type

        self.log = LogMock()
        self.data_interface = DataInterface(species, window_size=100, make_new=False,
            download_if_not_exists=False, log=self.log, load_genes=True,
            path = '.emptyh5.h5')

        self.num_regions_supplied = len(regions)

        regions = self._check_region_specification(regions)

        self.region_set = genome_tools.RegionSet(regions, self.data_interface.genome)
        self.region_score_map = np.array([r.annotation for r in self.region_set.regions])

        self.rp_map = self._get_region_distances(
            self.data_interface.gene_loc_set, self.region_set, self.rp_decay
        )

        self.rp_map, self.refseq_genes, self.peaks = self.get_rp_map()

        if genes is None:
            genes = self.expression_model.genes

        self.gene_models = []
        for symbol in genes:
            try:
                self.gene_models.append(self._get_gene_model(symbol))
            except BadGeneException as err:
                logger.warn(str(err))


    def _get_region_distances(self, gene_loc_set, region_set, decay):

        #make regions x exons map and exons x genes map
        #try:
        indptr, indices, exons = [0],[],[]
        for locus in gene_loc_set.regions:
            new_exons = (locus.annotation.get_exon_regions()[0],)
            exons.extend(new_exons)
            indices.extend(range(indptr[-1], indptr[-1] + len(new_exons)))
            indptr.append(indptr[-1] + len(new_exons))

        exon_gene_map = sparse.csc_matrix((np.ones(len(exons)), indices, indptr), shape = (len(exons), len(gene_loc_set.regions)))
        
        exons = genome_tools.RegionSet(exons, self.data_interface.genome)
        region_exon_map = region_set.map_intersects(exons, distance_function = lambda x,y : x.overlaps(y, min_overlap_proportion=0.4),slop_distance=0) #REGIONS X EXONS

        region_exon_map = region_exon_map.dot(exon_gene_map).astype(np.bool)

        not_exon_promoter = 1 - region_exon_map.sum(axis = 1).astype(np.bool)

        basic_rp_map = self.data_interface._make_basic_rp_map(gene_loc_set, region_set, decay).transpose()

        enhanced_rp_map = basic_rp_map.multiply(not_exon_promoter) + basic_rp_map.multiply(region_exon_map)

        return enhanced_rp_map.transpose()

        #except Exception as err:
        #    print(repr(err))
            #return region_exon_map, exon_gene_map

    def save(self, prefix):
        for model in self.gene_models:
            model.save(prefix)

    def load(self, prefix):
        for model in self.gene_models:
            model.load(prefix)

    def get_allowed_genes(self):
        return [x[3] for x in self.refseq_genes]

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

    def _get_gene_model(self, gene_symbol):

        try:
            gene_idx = np.argwhere(np.array([x[3] for x in self.refseq_genes]) == gene_symbol)[0]
        except IndexError:
            raise BadGeneException('Gene {} not in RefSeq database for this species'.format(gene_symbol))

        region_mask = self.rp_map[gene_idx, :].tocsr().indices

        try:
            region_distances = -self.rp_decay * np.log2(np.array(self.rp_map[gene_idx, region_mask])).reshape(-1)
        except TypeError:
            raise BadGeneException('No adjacent peaks to gene {}'.format(gene_symbol))

        tss_start = self.refseq_genes[gene_idx[0]][1]
        strand = self.refseq_genes[gene_idx[0]][4]

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

        cis_model = None
        try:
            cis_model = self.cis_models.get_model(gene_symbol)
        except (AttributeError, KeyError):
            pass

        return PyroRPVI(gene_symbol, 
            origin = self.refseq_genes[gene_idx[0]],
            region_distances = region_distances,
            upstream_mask = upstream_mask,
            downstream_mask = downstream_mask,
            region_score_map = self.region_score_map,
            region_mask = region_mask,
            model_type = self.model_type,
            use_SGD = self.use_SGD,
            parent_model = cis_model
        )

    def add_accessibility_params(self, accessibility, latent_compositions):
        logger.debug('Adding peak accessibility to models ...')

        latent_compositions = _get_latent_vars(self.accessibility_model, matrix = accessibility,
            latent_comp= latent_compositions)

        for model in self.gene_models:
            model.clear_features()

        for imputed_peaks in self.accessibility_model._batch_impute(latent_compositions, bar = False):
        #for start, end in self.accessibility_model._iterate_batch_idx(accessibility.shape[0], 512):
        #    batch_peaks = accessibility[start : end]
        #    batch_peaks.data = np.ones_like(batch_peaks.data)
        #    batch_peaks = np.array(batch_peaks.todense()).astype(float)
            for model in self.gene_models:
                model.add_accessibility_params(imputed_peaks * 1e4)
                
        for model in self.gene_models:
            model.trans_features = model.to_tensor(latent_compositions)


    def add_expression_params(self, raw_expression):
        logger.debug('Adding modeled expression data to models ...')
        raw_expression = self.expression_model._validate_data(raw_expression)
        params = self.expression_model._get_expression_distribution_parameters(raw_expression)

        for model in self.gene_models:
            try:
                gene_idx = np.argwhere(self.expression_model.genes == model.gene)[0]
            except IndexError:
                raise Exception('Gene {} not modeled using expression model!'.format(model.gene))
            model.add_expression_params(raw_expression[:,gene_idx], *params)

    def set_fit_params(self,*, max_iters, learning_rate, batch_size, decay, momentum):
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay = decay
        self.momentum = momentum

    def get_fit_params(self):
        if not self.use_SGD:
            return dict(learning_rate = self.learning_rate, max_iters = self.max_iters)
        else:
            return dict(batch_size = self.batch_size, momentum = self.momentum,
                learning_rate = self.learning_rate, decay = self.decay, test_size = self.test_size,
                restarts = self.restarts, max_iters = self.max_iters)

    def get_model(self, gene):

        model_idxs = dict(zip(self.get_modeled_genes(), np.arange(len(self.get_modeled_genes()))))
        model_idx = model_idxs[gene]

        return self.gene_models[model_idx]


    @register_input
    def tune_hyperparameters(self, cv = KFold(5), tune_iters = 50,
        num_trial_models = 50, verbose = 2):

        N = len(self.gene_models[0])

        candidate_genes = np.intersect1d(self.expression_model.genes[self.expression_model.highly_variable], self.get_modeled_genes())
        trial_genes = np.random.choice(candidate_genes, size = num_trial_models, replace = False)

        trial_models = [self.get_model(gene) for gene in trial_genes]

        class ModelMapper(CisModeler, BaseEstimator):

            def __init__(self, expression_model, accessibility_model, gene_models, *, use_SGD = False,
                learning_rate = 0.1, decay = 0.95, batch_size = 64, max_iters = 200, momentum = 0.1, test_size = 0.2,
                restarts = 5):
                self.gene_models = gene_models
                self.expression_model = expression_model
                self.accessibility_model = accessibility_model
                self.learning_rate = learning_rate
                self.decay = decay
                self.batch_size = batch_size
                self.max_iters = max_iters
                self.momentum = momentum
                self.test_size = test_size
                self.restarts = restarts
                self.use_SGD = use_SGD

            def fit(self, train_idx):
                fit = []
                for model in self.gene_models:
                    try:
                        model.fit(train_idx=train_idx, **self.get_fit_params())
                        model.score()
                        fit.append(True)
                    except ModelingError:
                        logging.exception('{} model could not be trained'.format(model.gene))
                        fit.append(False)

                self.gene_models = [model for was_fit, model in zip(fit, self.gene_models)
                                    if was_fit]

                return self

            def score(self, test_idx):
                return np.mean([-model.score(test_idx)
                    for model in self.gene_models])

        mapper = ModelMapper(self.expression_model, self.accessibility_model, trial_models, use_SGD = self.use_SGD,
            test_size = self.test_size, restarts = self.restarts)
        params = self._get_param_space()
        self.searcher = RandomizedSearchCV(mapper, params, cv = cv, n_iter = tune_iters, refit = False,
            verbose = verbose)

        try:
            self.searcher.fit(np.arange(N))
        except KeyboardInterrupt:
            pass

        self.best_params = self.searcher.best_params_
        for k,v in self.best_params.items():
            self.__dict__[k] = v

        return self

    def _get_param_space(self):
        if self.use_SGD:
            return dict(
                batch_size = [32,64,128,256,512],
                learning_rate = stats.loguniform(1e-5, 0.01),
                decay = stats.loguniform(0.9, 1.0),
                momentum = stats.loguniform(0.5, 0.999),
            )
        else:
            return dict(
                learning_rate = stats.loguniform(0.0001, 0.1)
            )

    @register_input
    def load_features(self, idx = None):
        pass

    @register_input
    def fit(self, idx = None, bar = True):

        logger.debug('Training RP models ...')
        self.was_fit = []
        for model in tqdm.tqdm(self.gene_models, desc = 'Training models', disable = not bar):
            try:
                model.fit(idx = idx, **self.get_fit_params())
                model.score()
                self.was_fit.append(True)
            except (ModelingError, ValueError):
                logging.exception('{} model could not be trained'.format(model.gene))
                self.was_fit.append(False)

        #self.gene_models = [model for was_fit, model in zip(fit, self.gene_models)
        #                    if was_fit]

        return self

    @register_input
    def predict(self, idx = None):
        
        return np.hstack([
            model.predict(idx = idx) for model in tqdm.tqdm(self.gene_models, desc = 'Predicting expression')
        ])

    def get_modeled_genes(self):
        return np.array([m.gene for m in self.gene_models])

    @property
    def genes(self):
        return self.get_modeled_genes()


    @register_input
    def get_logp(self, idx = None, bar = True):
        return np.hstack(list(
            map(lambda m : m._get_logp(idx = idx), 
            tqdm.tqdm(self.gene_models, desc = 'Scoring models', disable = not bar)))
            )

    @register_input
    def score(self, idx = None, bar = True):
        return -self.get_logp(idx = idx, bar = bar).mean()

    def cis_trans_test(self, compare_models, pval_threshold = 0.01, idx = None,
        **feature_kwargs):
        assert(self.model_type == 'trans'), 'To call cis/trans test, must have trained with "model_type" set to "trans".'
        
        self.load_features(**feature_kwargs)
        compare_models.load_features(**feature_kwargs)

        dist = chi2(self.accessibility_model.num_topics)
        critical_val = dist.ppf(1 - pval_threshold)

        results = []
        for trans_model in self.gene_models:
            try:
                cis_model = compare_models.get_model(trans_model.gene)
                
                trans_logp = trans_model._get_logp(idx = idx).reshape(-1).sum()
                cis_logp = cis_model._get_logp(idx = idx).reshape(-1).sum()

                test_stat = 2*(trans_logp - cis_logp)

                results.append(dict(
                    gene = trans_model.gene,
                    significant = test_stat > critical_val,
                    pvalue = 1-dist.cdf(test_stat),
                    test_statistic = test_stat,
                    trans_logp = trans_logp,
                    cis_logp = cis_logp,
                ))
            except KeyError:
                pass

        return sorted(results, key = lambda x : -x['test_statistic'])

    @register_input
    def driver_TF_test(self, idx = None):

        hit_matrix = self.accessibility_model.factor_hits[:, self.region_score_map]

        delta_logp = []
        for model in tqdm.tqdm(self.gene_models, desc = 'Predicting driver TFs'):

            model_mask = hit_matrix[:, model.region_mask].toarray().astype(np.bool)
            delta_logp.append(model.prob_ISD(model_mask).reshape((-1,1)))

        delta_logp = np.hstack(delta_logp)

        delta_logp/=np.linalg.norm(delta_logp, ord = 2, axis = 0, keepdims = True)
        delta_logp-=np.mean(delta_logp, axis = 1, keepdims = True)

        return ProbISD_Results(
            factor_names = self.accessibility_model.factor_names, 
            factor_ids = self.accessibility_model.factor_ids,
            isd_score = delta_logp.T, 
            genes = self.genes, 
            accessibility_model = self.accessibility_model, 
            expression_model = self.expression_model,
        )


class PyroRPVI(PyroModule):
    
    def __init__(self, gene_name, *, origin, region_distances, upstream_mask, downstream_mask, 
        region_score_map, region_mask, use_cuda = False, model_type = 'cis', use_SGD = False,
        parent_model = None):
        super().__init__()

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logger.debug('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.to(self.device)

        self.origin = origin
        self.region_score_map = region_score_map
        self.region_mask = region_mask
        self.gene = gene_name
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = ~np.logical_or(self.upstream_mask, self.downstream_mask)
        self.model_type = model_type

        if model_type == 'cis':
            self.forward = self.cis_model
        elif model_type == 'trans':
            self.forward = self.trans_model
        else:
            raise Exception('Invalid model type')
        
        if not parent_model is None:
            self.guide_params = {k.replace('cis_', self.model_type + '_') : v for
                k, v in parent_model.guide().items()}
            self.bn_mean_init = parent_model.bn.running_mean
            self.bn_var_init = parent_model.bn.running_var

        self.upstream_distances = self.to_tensor(region_distances[np.newaxis, self.upstream_mask])
        self.downstream_distances = self.to_tensor(region_distances[np.newaxis, self.downstream_mask])
        
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)
        self.clear_features()

        if use_SGD:
            self.fit = self.fit_SGD

    def get_normalized_params(self):
        model_params = {}
        for k, v in self.guide().items():
            v = v.detach()
            if len(v.size()) > 0:
                v = list(map(float, np.squeeze(v.numpy())))
            else:
                v = float(v)
            k = k.split('/')[-1].strip(self.gene)
            model_params[k]=v
            
        return model_params

    def get_bounds(self, extend = 5):

        model_params = self.get_normalized_params()

        chrom, start, end, _, strand = self.origin
        up_decay, down_decay = model_params['logdistance']

        def scale_dist(extend, decay):
            return int(1500 + extend * 1e3 * decay)

        if strand == '+':
            return (
                chrom, 
                str(int(start) - scale_dist(extend, up_decay)),
                str(int(end) + scale_dist(extend, down_decay))
            )
        else:
            return (
                chrom, 
                str(int(start) - scale_dist(extend, down_decay)),
                str(int(end) + scale_dist(extend, up_decay))
            )

    def clear_features(self):
        self.upstream_weights = []
        self.downstream_weights = []
        self.promoter_weights = []
        self.trans_features = None

    def has_features(self):
        return len(self.upstream_weights) > 0

    def add_accessibility_params(self, region_weights):

        region_weights = region_weights[:, self.region_score_map[self.region_mask]]# * 1e4

        self.upstream_weights.append(region_weights[:, self.upstream_mask])
        self.downstream_weights.append(region_weights[:, self.downstream_mask])
        self.promoter_weights.append(region_weights[:, self.promoter_mask])

    def add_expression_params(self, raw_expression, rd_loc, rd_scale, softmax_denom):

        self.gene_expr = self.to_tensor(raw_expression)
        self.rd_loc = self.to_tensor(rd_loc.reshape(-1))
        self.rd_scale = self.to_tensor(rd_scale.reshape(-1))
        self.softmax_denom = self.to_tensor(softmax_denom)

    def to_tensor(self, x, requires_grad = False):
        return torch.tensor(x, requires_grad = False).to(self.device)

    def RP(self, weights, distances, d):
        return (weights * torch.pow(0.5, distances/(1e3 * d))).sum(-1)

    def get_NB_distribution(self, activation, ind, theta, 
        gamma, bias, deterministic = True):

        return get_expression_distribution(activation.reshape((-1,1)),
                batchnorm = self.bn, 
                rd_loc = self.rd_loc.index_select(0, ind).reshape((-1,1)), 
                rd_scale = self.rd_scale.index_select(0, ind).reshape((-1,1)), 
                softmax_denom = self.softmax_denom.index_select(0, ind).reshape((-1,1)), 
                dispersion = theta, gamma = gamma, bias = bias, 
                deterministic=deterministic)

    def get_prefix(self):
        return self.model_type + '_' + self.gene
    
    def cis_model(self, batch_size = None, idx = None):

        with scope(prefix = self.get_prefix()):
            if idx is None:
                idx = np.arange(len(self.rd_loc))

            with pyro.plate("regions", 3):
                a = pyro.sample("a", dist.HalfNormal(1.))

            with pyro.plate("upstream-downstream", 2):
                d = pyro.sample('logdistance', dist.LogNormal(np.e, 1.))

            theta = pyro.sample('theta', dist.Gamma(2., 0.5))
            gamma = pyro.sample('gamma', dist.LogNormal(0., 0.5))
            bias = pyro.sample('bias', dist.Normal(0, 5.))

            with pyro.plate('data', len(idx), subsample_size=batch_size) as ind:

                ind = torch.tensor(idx).index_select(0, ind)

                f_Z = a[0] * self.RP(self.upstream_weights.index_select(0, ind), self.upstream_distances, d[0])\
                    + a[1] * self.RP(self.downstream_weights.index_select(0, ind), self.downstream_distances, d[1]) \
                    + a[2] * self.promoter_weights.index_select(0, ind).sum(-1)
                
                pyro.deterministic('f_Z', f_Z)
                NB = self.get_NB_distribution(f_Z, ind, theta, gamma, bias)
                pyro.sample('obs', NB.to_event(1), obs = self.gene_expr.index_select(0, ind))


    def trans_model(self, batch_size = None, idx = None):
        
        with scope(prefix = self.get_prefix()):
            if idx is None:
                idx = np.arange(len(self.rd_loc))

            with pyro.plate("regions", 3):
                a = pyro.sample('a', dist.HalfNormal(1.))

            with pyro.plate("upstream-downstream", 2):
                d = pyro.sample('logdistance', dist.LogNormal(np.e, 1.))

            theta = pyro.sample('theta', dist.Gamma(2., 0.5))
            gamma = pyro.sample('gamma', dist.LogNormal(0., 0.5))
            bias = pyro.sample('bias', dist.Normal(0, 5.))

            with pyro.plate('trans_coefs', self.trans_features.shape[-1]):
                a_trans = pyro.sample('a_trans', dist.Normal(0.,1.))

            with pyro.plate('data', len(idx), subsample_size=batch_size) as ind:

                ind = torch.tensor(idx).index_select(0, ind)

                f_Z = a[0] * self.RP(self.upstream_weights.index_select(0, ind), self.upstream_distances, d[0])\
                    + a[1] * self.RP(self.downstream_weights.index_select(0, ind), self.downstream_distances, d[1]) \
                    + a[2] * self.promoter_weights.index_select(0, ind).sum(-1) + torch.matmul(self.trans_features.index_select(0, ind), torch.unsqueeze(a_trans, 0).T).reshape(-1)
                
                pyro.deterministic('f_Z', f_Z)
                NB = self.get_NB_distribution(f_Z, ind, theta, gamma, bias)
                pyro.sample('obs', NB.to_event(1), obs = self.gene_expr.index_select(0, ind))


    def _fix_features(self):

        if isinstance(self.upstream_weights, list):
            self.upstream_weights = self.to_tensor(np.vstack(self.upstream_weights))
            self.downstream_weights = self.to_tensor(np.vstack(self.downstream_weights))
            self.promoter_weights = self.to_tensor(np.vstack(self.promoter_weights))

    def sample_posterior(self, site, idx = None):

        if idx is None:
            idx = np.arange(self.get_num_obs())

        self._fix_features()
        self.eval()
        
        guide_trace = poutine.trace(self.guide).get_trace(idx = idx)
        model_trace = poutine.trace(poutine.replay(self.forward, guide_trace))\
            .get_trace(idx = idx)

        return model_trace.nodes[site]['value']

    def predict(self, idx=None):

        prediction = self.sample_posterior(self.get_prefix() + '/f_Z', idx = idx)
        return np.squeeze(self.to_numpy(prediction).T)[:, np.newaxis]

    def get_num_obs(self):
        return len(self.rd_loc)

    def _get_logp(self, idx = None):

        self._fix_features()

        f_Z = self.sample_posterior(self.get_prefix() + '/f_Z', idx = idx)
        theta, gamma, bias = [self.guide()[self.get_prefix() + '/' + param]
            for param in ['theta','gamma','bias']]

        if idx is None:
            idx = torch.tensor(np.arange(len(self)), requires_grad = False)
        else:
            idx = torch.tensor(idx, requires_grad = False)

        NB = self.get_NB_distribution(f_Z, idx, theta, gamma, bias, deterministic=True)

        return self.to_numpy(NB.log_prob(self.gene_expr.index_select(0, idx)))

    def score(self, idx = None):
        return -self._get_logp(idx).mean()

    def _get_weights(self):
        pyro.clear_param_store()
        self.bn = torch.nn.BatchNorm1d(1, momentum = 1.0, affine = False).double()

        try:
            self.guide_params
            self.guide = AutoDelta(self, 
                init_loc_fn = init_to_value(
                    values = {k : v.clone().detach().requires_grad_(True) for k, v in self.guide_params.items()},
                    fallback=init_to_mean)
                )
            self.bn.running_mean = self.bn_mean_init.clone().detach()
            self.bn.running_var = self.bn_var_init.clone().detach()

        except AttributeError:
            self.guide = AutoDelta(self, init_loc_fn = init_to_mean)

    @staticmethod
    def get_loss_fn():
        return TraceMeanField_ELBO().differentiable_loss

    def fit_SGD(self, train_idx = None, test_idx = None, test_size = 0.2, 
        learning_rate = 0.001, max_iters = 200, batch_size = 64, decay = 0.95, patience = 5, momentum = 0.1, restarts = 5):

        if train_idx is None:
            train_idx = np.arange(len(self))

        if test_idx is None:
            test_mask = np.random.rand(len(train_idx)) < test_size
            test_idx = train_idx[test_mask]
            train_idx = train_idx[~test_mask]
            
        scores, weights, self.test_losses = [],[],[]
        for restart in range(restarts):
            try:
                self._fit_SGD_step(train_idx, test_idx, learning_rate = learning_rate, max_iters = max_iters, batch_size = batch_size, 
                    decay =decay, patience = patience, momentum = momentum)
                scores.append(self.testing_loss[-1])
                weights.append(self._get_save_data())
                self.test_losses.append(self.testing_loss)
            except ValueError:
                learning_rate*=0.5
                logging.warn('Error training {} model'.format(self.gene))

        if len(scores) == 0:
            raise ValueError('Model {} could not be fit!'.format(self.gene))

        best_model = np.argmin(scores)
        self._load_save_data(weights[best_model])

        return self

    def _fit_SGD_step(self, train_idx, test_idx, learning_rate = 0.001, max_iters = 200,
        batch_size = 64, decay = 0.95, patience = 5, momentum = 0.1):
        
        pyro.clear_param_store()
        self._get_weights()

        N = len(self)
        iters_per_epoch = N//batch_size
        min_iters = max(5, iters_per_epoch)

        lr_function = partial(_warmup_lr_decay, n_batches_per_epoch = iters_per_epoch, lr_decay = decay)
        scheduler = pyro.optim.LambdaLR({'optimizer': SGD, 'optim_args': {'lr': learning_rate, 'momentum' : momentum}, 
            'lr_lambda' : lambda e : lr_function(e)})

        svi = SVI(self, self.guide, scheduler, loss=TraceMeanField_ELBO())
        early_stopper = EarlyStopping(patience = patience, tolerance = 1e-4)

        self._fix_features()

        self.testing_loss = []
        for j in range(int(max_iters)):
            running_loss = 0

            self.train()
            running_loss+= float(svi.step(batch_size, idx = train_idx))/len(test_idx)
            scheduler.step()

            self.eval()
            self.testing_loss.append(
                float(svi.evaluate_loss(None, idx = test_idx))/len(test_idx)
            )

            if j > min_iters and early_stopper.should_stop_training(self.testing_loss[-1]):
                logger.debug('Stopped training early!')
                break

        self.num_iters_trained = j
        self.posterior_mean = self.guide.median()
        return self

    def fit(self, idx = None, learning_rate = 0.1, max_iters = 200):

        self.attempts = 0
        while True:
            try:
                pyro.clear_param_store()
                self._get_weights()
                self._fix_features()
                N = len(self)

                loss_fn = self.get_loss_fn()

                with poutine.trace(param_only=True) as param_capture:
                    loss = loss_fn(self.forward, self.guide, batch_size = None, idx = idx)

                params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}
                optimizer = torch.optim.LBFGS(params, lr=learning_rate)
                early_stopper = EarlyStopping(patience = 3, tolerance = 1e-3)

                def closure():
                    optimizer.zero_grad()
                    loss = loss_fn(self.forward, self.guide, batch_size = None, idx = idx)
                    loss.backward()
                    return loss

                self.testing_loss = []
                for iternum in range(max_iters):
                    self.testing_loss.append(self.to_numpy(optimizer.step(closure))/len(self))

                    if iternum > 2 and self.testing_loss[-1] > self.testing_loss[0]:
                        raise ValueError()

                    if early_stopper.should_stop_training(self.testing_loss[-1]):
                        break

            except ValueError:
                learning_rate*=0.5
                self.attempts+=1
                if self.attempts >= 10:
                    raise ModelingError('Model {} could not be fit.'.format(self.gene))
            else:
                break

        return self


    def get_savename(self, prefix):
        return prefix + self.gene + '.pth'

    def _get_save_data(self):
        return dict(bn = self.bn.state_dict(), guide = self.guide)

    def _load_save_data(self, state):
        self._get_weights()
        self.bn.load_state_dict(state['bn'])
        self.guide = state['guide']

    def save(self, prefix):
        #torch.save(self.state_dict(), self.get_savename(prefix))
        torch.save(self._get_save_data(), self.get_savename(prefix))

    def load(self, prefix):
        #self.load_state_dict(torch.load(self.get_savename(prefix)))
        state = torch.load(self.get_savename(prefix))
        self._load_save_data(state)

    def __len__(self):
        return len(self.rd_loc)
    
    @staticmethod
    def to_numpy(X):
        return X.detach().cpu().numpy()

    def prob_ISD(self, hit_masks):

        running_mean, running_var = self.bn.running_mean.clone().detach(), self.bn.running_var.clone().detach()
        upstream_weights = self.upstream_weights.clone().detach()
        promoter_weights = self.promoter_weights.clone().detach()
        downstream_weights = self.downstream_weights.clone().detach()

        init_logp = self.get_logp().sum()

        try:
            
            logps = []
            for hit_mask in hit_masks:
                self.upstream_weights = upstream_weights.clone()
                self.upstream_weights[:, hit_mask[self.upstream_mask]] = 0.

                self.downstream_weights = downstream_weights.clone()
                self.downstream_weights[:, hit_mask[self.downstream_mask]] = 0.

                self.promoter_weights = promoter_weights.clone()
                self.promoter_weights[:, hit_mask[self.promoter_weights]] = 0.

                logps.append(self.get_logp().sum())

        finally:
            self.running_mean = running_mean
            self.running_var = running_var

        return init_logp - np.array(logps)


    def prob_ISD(self, hit_masks):

        num_factors, num_regions = hit_masks.shape
        hit_masks = np.concatenate([np.zeros((1, num_regions)).astype(bool), hit_masks]) #factors, regions

        def detach_and_tile(x):
            x = x.clone().detach().numpy()
            x = np.expand_dims(x, -1)
            return np.tile(x, num_factors+1).transpose((0,2,1))

        def detach(x):
            return x.clone().detach().numpy()

        def delete_regions(weights, region_mask):
            hits = 1 - hit_masks[:, region_mask][np.newaxis, :, :] #1, factors, regions
            return np.multiply(weights, hits)

        upstream_weights = delete_regions(detach_and_tile(self.upstream_weights), self.upstream_mask) #cells, factors, regions
        promoter_weights = delete_regions(detach_and_tile(self.promoter_weights), self.promoter_mask)
        downstream_weights = delete_regions(detach_and_tile(self.downstream_weights), self.downstream_mask)

        rd_loc = detach(self.rd_loc)[:, np.newaxis]
        softmax_denom = detach(self.softmax_denom)[:, np.newaxis]
        expression = detach(self.gene_expr)
        
        upstream_distances = detach(self.upstream_distances)[:, np.newaxis, :]
        downstream_distances = detach(self.downstream_distances)[:,np.newaxis, :]

        def RP(weights, distances, d):
            return (weights * np.power(0.5, distances/(1e3 * d))).sum(-1)

        params = self.get_normalized_params()

        f_Z = params['a'][0] * RP(upstream_weights, upstream_distances, params['logdistance'][0]) \
        + params['a'][1] * RP(downstream_weights, downstream_distances, params['logdistance'][1]) \
        + params['a'][2] * promoter_weights.sum(-1) # cells, factors
        
        #f_Z = (f_Z - self.bn.running_mean.numpy())/(np.sqrt(self.bn.running_var.numpy() + self.bn.eps))
        f_Z = (f_Z - f_Z.mean(0,keepdims = True))/np.sqrt(f_Z.var(0, keepdims = True) + self.bn.eps)
        #f_Z = (f_Z - f_Z.mean())/np.sqrt(f_Z.var() + self.bn.eps)

        indep_rate = np.exp(params['gamma'] * f_Z + params['bias'])
        compositional_rate = indep_rate/softmax_denom
        
        mu = np.exp(rd_loc) * compositional_rate
        
        p = mu / (mu + params['theta'])
        
        logp_data = nbinom(params['theta'], 1 - p).logpmf(expression).sum(0)
        
        return logp_data[0] - logp_data[1:]
