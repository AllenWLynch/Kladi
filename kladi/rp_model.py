
import torch
import logging
import tqdm
import pyro
from pyro.nn.module import PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta
import pyro.distributions as dist
#from pyro.optim import Adam
from torch.optim import Adam, SGD
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_sample, init_to_mean
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
from sklearn.model_selection import ParameterSampler, KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator
from functools import partial

logger = logging.getLogger(__name__)

class LogMock:

    def __init__(self):
        pass

    def append(self, *args, **kwargs):
        logger.debug(*args)

class BadGeneException(Exception):
    pass


def _warmup_lr_decay(step,*,n_batches_per_epoch, lr_decay):
    return min(1., step/n_batches_per_epoch)*(lr_decay**int(step/n_batches_per_epoch))


def _get_latent_vars(model, latent_comp = None, matrix = None):

    assert(not latent_comp is None or not matrix is None)

    if latent_comp is None:
        latent_compositions = model.predict(matrix)
    else:
        latent_compositions = latent_comp
        model._check_latent_vars(latent_compositions)

    return latent_compositions.copy()

def get_expression_distribution(f_Z,*, batchnorm, rd_loc, rd_scale, softmax_denom, 
    dispersion, deterministic = False):

    #print(f_Z.shape, rd_loc.shape, rd_scale.shape, softmax_denom.shape, dispersion.shape)
    if not deterministic:
        read_depth = torch.distributions.LogNormal(rd_loc, rd_scale).sample()
    else:
        read_depth = rd_loc.exp()

    rate =  batchnorm(f_Z).exp()/softmax_denom
    mu = read_depth * rate

    p = torch.minimum(mu / (mu + dispersion), torch.tensor([0.99999], requires_grad = False))

    return dist.NegativeBinomial(total_count = dispersion, probs = p)


class CisModeler(FromRegions):

    rp_decay = 60000

    def __init__(self, species,*,accessibility_model, expression_model, genes = None, 
        max_iters = 150, learning_rate = 0.001, batch_size = 64, decay = 0.95, momentum = 0.1):

        assert(species in ['mm10','hg38'])
        regions = accessibility_model.peaks.tolist()
        assert(isinstance(regions, (list, tuple))), '"regions" parameter must be list of region tuples in format [ (chr,start,end [,score]), (chr,start,end [,score]) ... ] or name of bed file.'
        
        self.accessibility_model = accessibility_model
        self.expression_model = expression_model

        self.set_fit_params(max_iters = max_iters, learning_rate = learning_rate,
            batch_size = batch_size, decay = decay, momentum = momentum)

        self.log = LogMock()
        self.data_interface = DataInterface(species, window_size=100, make_new=False,
            download_if_not_exists=False, log=self.log, load_genes=True,
            path = '.emptyh5.h5')

        self.num_regions_supplied = len(regions)

        regions = self._check_region_specification(regions)

        self.region_set = genome_tools.RegionSet(regions, self.data_interface.genome)
        self.region_score_map = np.array([r.annotation for r in self.region_set.regions])

        self.rp_map = 'basic'
        self.rp_map, self.refseq_genes, self.peaks = self.get_rp_map()

        if genes is None:
            genes = self.expression_model.genes

        self.gene_models = []
        for symbol in genes:
            try:
                self.gene_models.append(self._get_gene_model(symbol))
            except BadGeneException as err:
                logger.warn(str(err))

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

        return PyroRPVI(gene_symbol, 
            region_distances = region_distances,
            upstream_mask = upstream_mask,
            downstream_mask = downstream_mask,
            region_score_map = self.region_score_map,
            region_mask = region_mask
        )

    def add_accessibility_params(self, accessibility = None, latent_compositions = None):
        logger.debug('Adding peak accessibility to models ...')

        latent_compositions = _get_latent_vars(self.accessibility_model, matrix = accessibility,
            latent_comp= latent_compositions)

        for model in self.gene_models:
            model.clear_features()

        for imputed_peaks in self.accessibility_model._batch_impute(latent_compositions, bar = False):
            for model in self.gene_models:
                model.add_accessibility_params(imputed_peaks)

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
        return dict(max_iters = self.max_iters, learning_rate = self.learning_rate,
            batch_size = self.batch_size, decay = self.decay, momentum = self.momentum)

    def get_model(self, gene):

        model_idxs = dict(zip(self.get_modeled_genes(), np.arange(len(self.get_modeled_genes()))))
        model_idx = model_idxs[gene]

        return self.gene_models[model_idx]

    def tune_hyperparameters(self, raw_expression, accessibility_matrix = None, 
        accessibility_latent_compositions = None, cv = KFold(5), tune_iters = 50,
        num_trial_models = 50, verbose = 2):

        latent_compositions = _get_latent_vars(self.accessibility_model, matrix = accessibility_matrix,
            latent_comp= accessibility_latent_compositions)

        self.add_accessibility_params(accessibility = None, 
            latent_compositions = latent_compositions)
        self.add_expression_params(raw_expression)

        candidate_genes = np.intersect1d(self.expression_model.genes[self.expression_model.highly_variable], self.get_modeled_genes())
        trial_genes = np.random.choice(candidate_genes, size = num_trial_models, replace = False)
        #trial_models = np.random.choice(self.gene_models, size= num_trial_models, replace = False)

        trial_models = [self.get_model(gene) for gene in trial_genes]

        class ModelMapper(CisModeler, BaseEstimator):

            def __init__(self, expression_model, accessibility_model, gene_models, *,learning_rate = 0.1, decay = 0.95, batch_size = 64, max_iters = 200, momentum = 0.1):
                self.gene_models = gene_models
                self.expression_model = expression_model
                self.accessibility_model = accessibility_model
                self.learning_rate = learning_rate
                self.decay = decay
                self.batch_size = batch_size
                self.max_iters = max_iters
                self.momentum = momentum

            def fit(self, X, y):
                return super().fit(X, accessibility_latent_compositions = y, bar = False)

            def score(self, X, y):
                return -super().score(X, accessibility_latent_compositions = y, bar = False)

        mapper = ModelMapper(self.expression_model, self.accessibility_model, trial_models)

        params = dict(
            batch_size = [32,64,128,256,512],
            max_iters = [200],
            learning_rate = stats.loguniform(1e-5, 0.01),
            decay = stats.loguniform(0.9, 1.0),
            momentum = stats.loguniform(0.1, 0.999),
        )

        self.searcher = RandomizedSearchCV(mapper, params, cv = cv, n_iter = tune_iters, refit = False, verbose = verbose)
        self.searcher.fit(raw_expression, y = latent_compositions)

        best_params = self.searcher.best_params_
        self.set_fit_params(**best_params)

        return best_params


    def fit(self,raw_expression, accessibility_matrix = None, 
        accessibility_latent_compositions = None, bar = True):

        self.add_accessibility_params(accessibility = accessibility_matrix, 
            latent_compositions = accessibility_latent_compositions)
        self.add_expression_params(raw_expression)

        logger.debug('Training RP models ...')
        for model in tqdm.tqdm(self.gene_models, desc = 'Training models', disable = not bar):
            params = self.get_fit_params()
            successfully_fit, attempts = False, 1
            while not successfully_fit:
                try:
                    model.fit(**params)
                    successfully_fit=True
                except ValueError:
                    if attempts < 5:
                        params['learning_rate']/=10
                        attempts+=1
                    else:
                        raise ValueError('Model {} could not be trained!'.format(model.gene))

        return self

    def predict(self,raw_expression, accessibility_matrix = None, accessibility_latent_compositions = None):

        self.add_accessibility_params(accessibility = accessibility_matrix, 
            latent_compositions = accessibility_latent_compositions)
        self.add_expression_params(raw_expression)
        
        return np.hstack([
            model.predict() for model in tqdm.tqdm(self.gene_models, desc = 'Predicting expression')
        ])

    def get_modeled_genes(self):
        return np.array([m.gene for m in self.gene_models])

    def _get_logp(self,raw_expression, accessibility_matrix = None, accessibility_latent_compositions = None, bar = True):

        self.add_accessibility_params(accessibility = accessibility_matrix, 
            latent_compositions = accessibility_latent_compositions)
        self.add_expression_params(raw_expression)

        return np.hstack(list(map(lambda m : m._get_logp(), tqdm.tqdm(self.gene_models, desc = 'Scoring models', disable = not bar))))


    def score(self, raw_expression, accessibility_matrix = None, accessibility_latent_compositions = None, bar = True):

        return -self._get_logp(raw_expression, accessibility_matrix = accessibility_matrix, 
            accessibility_latent_compositions = accessibility_latent_compositions, bar = bar).mean()


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
        self.gene = gene_name
        self.upstream_mask = upstream_mask
        self.downstream_mask = downstream_mask
        self.promoter_mask = ~np.logical_or(self.upstream_mask, self.downstream_mask)

        self.upstream_distances = self.to_tensor(region_distances[np.newaxis, self.upstream_mask])
        self.downstream_distances = self.to_tensor(region_distances[np.newaxis, self.downstream_mask])
        
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)
        self.clear_features()

        self.posterior_mean = torch.ones((5,))


    def clear_features(self):
        self.upstream_weights = []
        self.downstream_weights = []
        self.promoter_weights = []

    def add_accessibility_params(self, region_weights):

        region_weights = region_weights[:, self.region_score_map[self.region_mask]] * 1e4

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

    def get_NB_distribution(self, activation, ind, theta, deterministic = False):

        return get_expression_distribution(activation.reshape((-1,1)),
                batchnorm = self.bn, 
                rd_loc = self.rd_loc.index_select(0, ind).reshape((-1,1)), 
                rd_scale = self.rd_scale.index_select(0, ind).reshape((-1,1)), 
                softmax_denom = self.softmax_denom.index_select(0, ind).reshape((-1,1)), 
                dispersion = theta, deterministic=deterministic)
    
    def forward(self, batch_size = None, idx = None):

        if idx is None:
            idx = np.arange(len(self.rd_loc))

        pyro.module("batchnorm", self.bn)

        with pyro.plate(self.gene +"_regions", 3):
            a = pyro.sample(self.gene +"_a", dist.HalfNormal(12.))

        with pyro.plate(self.gene +"_upstream-downstream", 2):
            d = torch.exp(pyro.sample(self.gene +'_logdistance', dist.Normal(np.e, 2.)))

        theta = pyro.sample(self.gene +"_theta", dist.Gamma(2., 0.5))

        with pyro.plate(self.gene +"_data", len(idx), subsample_size=batch_size) as ind:

            ind = torch.tensor(idx).index_select(0, ind)

            activation = a[0] * self.RP(self.upstream_weights.index_select(0, ind), self.upstream_distances, d[0])\
                + a[1] * self.RP(self.downstream_weights.index_select(0, ind), self.downstream_distances, d[1]) \
                + a[2] * self.promoter_weights.index_select(0, ind).sum(-1)

            pyro.deterministic('activation', activation)

            NB = self.get_NB_distribution(activation, ind, theta)

            pyro.sample(self.gene + '_obs', NB.to_event(1), obs = self.gene_expr.index_select(0, ind))

    def _fix_features(self):

        if isinstance(self.upstream_weights, list):
            #make into tensors
            self.upstream_weights = self.to_tensor(np.vstack(self.upstream_weights))
            self.downstream_weights = self.to_tensor(np.vstack(self.downstream_weights))
            self.promoter_weights = self.to_tensor(np.vstack(self.promoter_weights))

    def rail_guide(self):
        return {self.gene + '_a' : torch.unsqueeze(self.posterior_mean[:3].exp(),0), 
                self.gene + '_logdistance' : torch.unsqueeze(self.posterior_mean[3:5], 0), 
                self.gene + '_theta' : self.posterior_mean[5].exp().reshape((1,1))}

    def predict(self):

        self._fix_features()
        self.eval()
        
        attempts, success = 0, False
        
        while not success and attempts < 10:
            try:
                prediction = Predictive(self, posterior_samples = self.rail_guide(), 
                    return_sites = ['activation'])(None)['activation']
                success = True
            except ValueError:
                attempts += 1
                if attempts > 10:
                    raise ValueError()

        return self.to_numpy(prediction).T

    def get_num_obs(self):
        return len(self.rd_loc)

    def _get_logp(self):

        self._fix_features()

        activation = self.predict()
        theta = self.rail_guide()[self.gene + '_theta'][0]
        ind = torch.tensor(np.arange(self.get_num_obs()))

        NB = self.get_NB_distribution(torch.tensor(activation), ind, theta, deterministic=True)

        return self.to_numpy(NB.log_prob(self.gene_expr))

    def score(self):
        return -self._get_logp().mean()

    def _get_weights(self):
        pyro.clear_param_store()
        self.bn = torch.nn.BatchNorm1d(1).double()
        self.guide = AutoDiagonalNormal(self, init_loc_fn = init_to_mean)

    def fit(self, learning_rate = 0.001, max_iters = 200,
        batch_size = 64, decay = 0.95, patience = 5, momentum = 0.1):
        
        self._get_weights()

        N = len(self.rd_loc)
        iters_per_epoch = N//batch_size
        min_iters = 10

        test_set = np.random.rand(N) < 0.2
        train_set = ~test_set
        test_set = np.argwhere(test_set)[:,0]
        train_set = np.argwhere(train_set)[:,0]
        
        lr_function = partial(_warmup_lr_decay, n_batches_per_epoch = iters_per_epoch, lr_decay = decay)
        scheduler = pyro.optim.LambdaLR({'optimizer': SGD, 'optim_args': {'lr': learning_rate, 'momentum' : momentum}, 
            'lr_lambda' : lambda e : lr_function(e)})
        #scheduler = pyro.optim.PyroOptim(torch.optim.LBFGS, {'lr' : learning_rate}, {})
        svi = SVI(self, self.guide, scheduler, loss=TraceMeanField_ELBO())
        early_stopper = EarlyStopping(patience = patience, tolerance = 1e-4)

        '''class MyModel(nn.Module):
            def __init__(self, ...):
                ...

            def model(self, batch):
                ....

            def guide(self, batch):
                ...

        model = MyModel(...)

        elbo = pyro.infer.Trace_ELBO()
        optim = torch.optim.SGD(model.parameters())  # or LBFGS etc
        for batch in data:
            optim.zero_grad()
            elbo.loss_and_grads(model.model, model.guide, batch)
            optim.step()'''

        self._fix_features()

        self.testing_loss = []
        for j in range(int(max_iters)):
            running_loss = 0

            self.train()
            running_loss+= float(svi.step(batch_size, idx = train_set))/N
            scheduler.step()

            self.eval()
            self.testing_loss.append(
                float(svi.evaluate_loss(None, idx = test_set))/test_set.sum()
            )

            #if j%iters_per_epoch == 0:
            #    self.training_loss.append(running_loss)
            if j > min_iters and early_stopper.should_stop_training(self.testing_loss[-1]):
                logger.debug('Stopped training early!')
                break

        self.num_iters_trained = j
        self.posterior_mean = self.guide.get_posterior().mean
        return self

    def get_savename(self, prefix):
        return prefix + self.gene + '.pth'

    def _get_save_data(self):
        return dict(bn = self.bn.state_dict(), posterior_mean = self.posterior_mean)

    def save(self, prefix):
        #torch.save(self.state_dict(), self.get_savename(prefix))
        torch.save(self._get_save_data(), self.get_savename(prefix))

    def load(self, prefix):
        #self.load_state_dict(torch.load(self.get_savename(prefix)))
        state = torch.load(self.get_savename(prefix))
        self.bn = torch.nn.BatchNorm1d(1).double()
        self.bn.load_state_dict(state['bn'])
        self.posterior_mean = state['posterior_mean']


    def __len__(self):
        return len(self.rd_loc)


    '''def get_MAP_estimator(self):
        
        params = {k : self.to_numpy(v) for k, v in self.state_dict()}
        guide_locs = params['guide.loc']

        return RPModelPointEstimator(
            self.gene,
            a_up = guide_locs[0], a_down = guide_locs[1], a_promoter = guide_locs[2],
            distance_up = guide_locs[3], distance_down = guide_locs[4],
            theta = guide_locs[5],
            upstream_mask = self.upstream_mask,
            downstream_mask = self.downstream_mask,
            promoter_mask  = self.promoter_mask,
            region_mask = self.region_mask, 
            region_score_map = self.region_score_map,
            upstream_distances = self.to_numpy(self.upstream_distances),
            downstream_distances = self.to_numpy(self.downstream_distances),
        )'''
    
    @staticmethod
    def to_numpy(X):
        return X.detach().cpu().numpy()

class TransModel(PyroModule, BaseEstimator):

    bn_keys = ['batchnorm.weight', 'batchnorm.bias', 
        'batchnorm.running_mean', 'batchnorm.running_var', 'batchnorm.num_batches_tracked']

    def __init__(self,*, accessibility_model, expression_model, dropout = 0.2, learning_rate = 0.1, batch_size = 64, 
        num_iters = 200, seed = None, decay = 0.95, momentum = 0.9):
        super().__init__()
        self.accessibility_model = accessibility_model
        self.expression_model = expression_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.dropout = dropout
        self.seed = seed
        self.decay = decay
        self.momentum = momentum

    def tune_hyperparameters(self,raw_expression, accessibility_matrix = None, 
        accessibility_latent_compositions = None, cv = KFold(5), tune_iters = 50, verbose = 2):

        atac_latent_compositions = _get_latent_vars(self.accessibility_model, 
            latent_comp=accessibility_latent_compositions, matrix=accessibility_matrix)

        param_distributions = dict(
            dropout = stats.uniform(0, 0.3),
            seed = stats.randint(1, 2**32-1),
            learning_rate = stats.loguniform(1e-6, 1e-3),
            batch_size = [32,64,128,256,512],
            decay = stats.loguniform(0.9, 1.0),
            momentum = stats.loguniform(0.1,0.999),
        )

        def _score(estimator, X, y):
            return -estimator.score(X, y)

        self.searcher = RandomizedSearchCV(self, param_distributions, n_iter = tune_iters, 
            verbose = verbose, refit = False, cv = cv, scoring = _score)
        self.searcher.fit(raw_expression, y = atac_latent_compositions, bar = False)

        best_params = self.searcher.best_params_
        self.set_params(**best_params)

        return best_params


    def _set_seeds(self):

        seed = self.seed
        if seed is None:
            seed = int(time.time() * 1e7)%(2**32-1)

        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

    def _get_weights(self):

        pyro.clear_param_store()
        self._set_seeds()

        self.beta = torch.nn.Linear(self.accessibility_model.num_topics, 
            self.expression_model.num_exog_features,
            bias = False)
        self.drop = torch.nn.Dropout(self.dropout)
        self.batchnorm = torch.nn.BatchNorm1d(self.expression_model.num_exog_features)
        self.guide = AutoDiagonalNormal(self, init_loc_fn = init_to_mean)

    def get_modeled_genes(self):
        return self.expression_model.genes

    def forward(self, expression, Z, rd_loc, rd_scale, softmax_denom, stochastic_batch = True):

        pyro.module("batchnorm", self.batchnorm)
        pyro.module("beta", self.beta)
        pyro.module("dropout", self.drop)

        with pyro.plate('gene_plate', self.expression_model.num_exog_features):
            theta = pyro.sample("theta", dist.Gamma(2., 0.5))

        activation = self.beta(self.drop(Z))

        pyro.deterministic('activation', activation)

        with pyro.plate('samples_plate', expression.shape[0], subsample_size = self.batch_size if stochastic_batch else None) as ind: 

            NB = get_expression_distribution(f_Z = activation.index_select(0, ind),
                batchnorm = self.batchnorm, 
                rd_loc = rd_loc.index_select(0, ind), 
                rd_scale = rd_scale.index_select(0, ind), 
                softmax_denom = softmax_denom.index_select(0, ind).reshape((-1,1)),
                dispersion = theta)

            pyro.sample('obs', NB.to_event(1), obs = expression.index_select(0, ind))


    def _featurize(self, raw_expression, atac_latent_compositions):
        rd_loc, rd_scale, softmax_denom = list(map(lambda x : torch.tensor(x, requires_grad = False), self.expression_model._get_expression_distribution_parameters(raw_expression)))

        raw_expression = self.expression_model._validate_data(raw_expression)

        raw_expression = torch.tensor(np.vstack([
            batch[0] for batch in self.expression_model._get_batches(raw_expression)
        ]), requires_grad = False)
        
        return raw_expression, torch.tensor(atac_latent_compositions), rd_loc, rd_scale, softmax_denom

    def save(self, filename):

        state_dict = {k : self.state_dict()[k] for k in 
            self.bn_keys + ['posterior_mean','beta.weight']}
        torch.save(state_dict, filename)

    def load(self, filename):

        self._get_weights()

        state_dict = torch.load(filename)
        
        self.beta.load_state_dict({'weight' : state_dict['beta.weight']})
        self.posterior_mean = state_dict['posterior_mean']
        self.batchnorm.load_state_dict({param.split('.')[1] : state_dict[param] for param in self.bn_keys})


    def predict(self,raw_expression, accessibility_matrix = None, accessibility_latent_compositions = None):
        
        atac_latent_compositions = _get_latent_vars(self.accessibility_model, 
            latent_comp=accessibility_latent_compositions, matrix=accessibility_matrix)

        expression, Z, rd_loc, rd_scale, softmax_denom = self._featurize(raw_expression, atac_latent_compositions)

        return self.beta(Z).detach().cpu().numpy()


    def _get_logp(self,raw_expression, accessibility_matrix = None, 
        accessibility_latent_compositions = None):

        atac_latent_compositions = _get_latent_vars(self.accessibility_model, 
            latent_comp=accessibility_latent_compositions, matrix=accessibility_matrix)

        expression, Z, rd_loc, rd_scale, softmax_denom = self._featurize(raw_expression, atac_latent_compositions)

        self.eval()

        NB = get_expression_distribution(f_Z=self.beta(Z), 
            batchnorm=self.batchnorm, rd_loc=rd_loc, rd_scale = rd_scale, 
            softmax_denom = softmax_denom.reshape((-1,1)), 
            dispersion = self.posterior_mean.exp())

        return NB.log_prob(expression).detach().cpu().numpy()


    def score(self, raw_expression, accessibility_latent_compositions = None, 
        accessibility_matrix = None):

        return -self._get_logp(raw_expression = raw_expression, accessibility_matrix = accessibility_matrix, 
            accessibility_latent_compositions = accessibility_latent_compositions).mean()

    
    def fit(self, raw_expression, accessibility_latent_compositions = None, 
        accessibility_matrix = None, bar = True):

        atac_latent_compositions = _get_latent_vars(self.accessibility_model, 
            latent_comp=accessibility_latent_compositions, matrix=accessibility_matrix)
        
        self._get_weights()

        N = len(atac_latent_compositions)
        iters_per_epoch = N//self.batch_size
        min_epochs = 20
        
        lr_function = partial(_warmup_lr_decay, n_batches_per_epoch = iters_per_epoch, lr_decay = self.decay)
        scheduler = pyro.optim.LambdaLR({'optimizer': SGD, 'optim_args': {'lr': self.learning_rate, 'momentum' : self.momentum}, 
            'lr_lambda' : lambda e : lr_function(e)})
        svi = SVI(self, self.guide, scheduler, loss=TraceMeanField_ELBO())
        early_stopper = EarlyStopping(patience = 5, tolerance = 1e-4)

        self.training_loss = []
        for j in tqdm.tqdm(range(int(self.num_iters)), desc = 'Training model', disable = not bar):

            self.train()
            self.training_loss.append(float(svi.step(
                    *self._featurize(raw_expression, atac_latent_compositions))) / (N * self.expression_model.num_exog_features)
            )       
            scheduler.step()
            
            if j > min_epochs and early_stopper.should_stop_training(self.training_loss[-1]):
                logger.debug('Stopped training early!')
                break

        self.posterior_mean = self.guide.get_posterior().mean
        return self


    def get_likelihood_difference(self, rp_modeler, raw_expression, accessibility_latent_compositions = None, 
        accessibility_matrix = None):
    
        score_kwargs = dict(raw_expression = raw_expression, 
                accessibility_matrix = accessibility_matrix, 
                accessibility_latent_compositions = accessibility_latent_compositions)

        cis_genes = rp_modeler.get_modeled_genes()
        trans_genes = self.expression_model.genes
        shared_genes = np.intersect1d(trans_genes, cis_genes)

        cis_genes_idx = dict(zip(cis_genes, np.arange(len(cis_genes))))
        trans_genes_idx = dict(zip(trans_genes, np.arange(len(trans_genes))))

        trans_idx, cis_idx = list(zip(*[
            (trans_genes_idx[gene], cis_genes_idx[gene]) 
            for gene in shared_genes
        ]))

        trans_logp = self._get_logp(**score_kwargs)[:, trans_idx]
        cis_logp = rp_modeler._get_logp(**score_kwargs)[:, cis_idx]
        
        return shared_genes, trans_logp, cis_logp

    def likelihood_ratio_test(degrees_of_freedom, shared_genes, trans_logp, cis_logp):

        test_statistic = (trans_logp.sum(0) - cis_logp.sum(0))
        return list(zip(shared_genes, test_statistic, stats.chi2(degrees_of_freedom).cdf(test_statistic)))


'''class RPModelPointEstimator:

    @classmethod
    def from_file(cls, filename):

        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)

        gene = state_dict['name']
        del state_dict['name']

        return cls(gene, **state_dict)

    def __init__(self, gene_symbol, *, a_up, a_down, a_promoter, distance_up, distance_down, b,
            upstream_mask, downstream_mask, promoter_mask, upstream_distances, downstream_distances,
            region_mask, region_score_map, theta = theta):

        self.gene = gene_symbol
        self.a_up = a_up
        self.a_down = a_down
        self.a_promoter = a_promoter
        self.distance_up = distance_up
        self.distance_down = distance_down
        self.theta = theta

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
        
        lam = upstream_effects + promoter_effects + downstream_effects

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
        ])'''