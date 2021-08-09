


from functools import partial
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm, trange
from pyro.nn import PyroModule, PyroParam
import numpy as np
import torch.distributions.constraints as constraints
import logging
from kladi.matrix_models.ilr import ilr, _gram_schmit_basis
from math import ceil
import time
from pyro.contrib.autoname import scope
from sklearn.base import BaseEstimator
from adata_interface import *

logger = logging.getLogger(__name__)

class EarlyStopping:

    def __init__(self, 
                 tolerance = 1e-4,
                 patience=3,
                 convergence_check = True):

        self.tolerance = tolerance
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.convergence_check = convergence_check

    def __call__(self, current_loss):
        
        if current_loss is None:
            pass
        else:
            if ((current_loss - self.best_loss) < -self.tolerance) or \
                (self.convergence_check and ((current_loss - self.best_loss) > 10*self.tolerance)):
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    return True
                self.wait += 1

            if current_loss < self.best_loss:
                self.best_loss = current_loss

        return False


class Decoder(nn.Module):
    
    def __init__(self,*,num_exog_features, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_exog_features, bias = False)
        self.bn = nn.BatchNorm1d(num_exog_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        return F.softmax(self.bn(self.beta(self.drop(inputs))), dim=1)

    def get_softmax_denom(self, inputs):
        return self.bn(self.beta(inputs)).exp().sum(-1)


class ModelParamError(ValueError):
    pass

class OneCycleLR_Wrapper(torch.optim.lr_scheduler.OneCycleLR):

    def __init__(self, optimizer, **kwargs):
        max_lr = kwargs.pop('max_lr')
        super().__init__(optimizer, max_lr, **kwargs)


def encoder_layer(input_dim, output_dim, nonlin = True, dropout = 0.2):
    layers = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim)]
    if nonlin:
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_fc_stack(layer_dims = [256, 128, 128, 128], dropout = 0.2, skip_nonlin = True):
    return nn.Sequential(*[
        encoder_layer(input_dim, output_dim, nonlin= not ((i >= (len(layer_dims) - 2)) and skip_nonlin), dropout = dropout)
        for i, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:]))
    ])


class BaseModel(nn.Module, BaseEstimator):

    I = 50

    def __init__(self,
            highly_variable = None,
            predict_expression = None,
            counts_layer = None,
            covariates = [],
            batch_index = None,
            num_topics = 16,
            hidden = 128,
            num_layers = 3,
            decoder_dropout = 0.2,
            encoder_dropout = 0.1,
            use_cuda = True,
            seed = None,
            min_learning_rate = 1e-6,
            max_learning_rate = 1e-1,
            beta = 0.95,
            batch_size = 64,
            ):
        super().__init__()

        self.highly_variable = highly_variable
        self.predict_expression = predict_expression
        self.counts_layer = counts_layer
        self.covariates = covariates
        self.batch_index = batch_index
        self.num_topics = num_topics
        self.hidden = hidden
        self.num_layers = num_layers
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.use_cuda = use_cuda
        self.seed = seed
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.beta = beta
        self.batch_size = batch_size

    def _set_seeds(self):
        if self.seed is None:
            seed = int(time.time() * 1e7)%(2**32-1)

        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

    def _get_weights(self):

        pyro.clear_param_store()
        torch.cuda.empty_cache()
        self._set_seeds()

        assert(isinstance(self.use_cuda, bool))
        assert(isinstance(self.num_modules, int) and self.num_modules > 0)
        assert(isinstance(self.features, (list, np.ndarray)))
        assert(len(self.features) == self.num_exog_features)

        self.use_cuda = torch.cuda.is_available() and use_cuda
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        if not use_cuda:
            logger.warn('Cuda unavailable. Will not use GPU speedup while training.')

        self.decoder = Decoder(
            num_exog_features=self.num_exog_features, 
            num_topics=self.num_topics, 
            dropout = self.decoder_dropout
        )

        self.encoder = self.encoder_model(
            num_endog_features = self.num_endog_features, 
            num_topics = self.num_topics, 
            hidden = self.hidden, 
            encoder_dropout = self.encoder_dropout, 
            num_layers = self.num_layers
        )

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(*self._get_gamma_moments(self.I, self.num_topics))

        self.pseudocount_mu = PyroParam(_counts_mu * torch.ones((self.num_topics,)), constraint = constraints.positive)

        self.pseudocount_std = PyroParam(np.sqrt(_counts_var) * torch.ones((self.num_topics,)), 
                constraint = constraints.positive)

        self.K = torch.tensor(self.num_topics, requires_grad = False)


    #,*, endog_features, exog_features, read_depth, anneal_factor = 1.
    def model(self):
        
        pyro.module("decoder", self.decoder)

        _alpha, _beta = self._get_gamma_parameters(self.I, self.num_topics)
        with pyro.plate("topics", self.num_topics):
            initial_counts = pyro.sample("a", dist.Gamma(self._to_tensor(_alpha), self._to_tensor(_beta)))

        theta_loc = self._get_prior_mu(initial_counts, self.K)
        theta_scale = self._get_prior_std(initial_counts, self.K)

        return theta_loc, theta_scale


    def guide(self):

        pyro.module("encoder", self.encoder)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(self.pseudocount_mu[k], self.pseudocount_std[k]))


    @staticmethod
    def _get_gamma_parameters(I, K):
        return 2., 2*K/I

    @staticmethod
    def _get_gamma_moments(I,K):
        return I/K, 0.5 * (I/K)**2

    @staticmethod
    def _get_lognormal_parameters_from_moments(m, v):
        m_squared = m**2
        mu = np.log(m_squared / np.sqrt(v + m_squared))
        var = np.log(v/m_squared + 1)

        return mu, var

    @staticmethod
    def _get_prior_mu(a, K):
        return a.log() - 1/K * torch.sum(a.log())

    @staticmethod
    def _get_prior_std(a, K):
        return torch.sqrt(1/a * (1-2/K) + 1/(K * a))

    @staticmethod
    def get_num_batches(N, batch_size):
        return N//batch_size + int(N % batch_size > 0)

    @staticmethod
    def _iterate_batch_idx(N, batch_size, bar = False, desc = None):

        num_batches = N//batch_size + int(N % batch_size > 0)
        for i in range(num_batches) if not bar else tqdm(range(num_batches), desc = desc):
            yield i * batch_size, (i + 1) * batch_size

    def _preprocess_endog(self, X):
        raise NotImplementedError()

    def _preprocess_exog(self, X):
        raise NotImplementedError()

    def _preprocess_read_depth(self, X):
        return self.to_tensor(X, requires_grad = False).to(self.device)

    def _iterate_batches(self, endog_features, exog_features, batch_size = 32, bar = True, desc = None):
        
        N = endog_features.shape[0]
        read_depth = np.array(exog_features.sum(-1)).reshape((-1,1))

        for start, end in self._iterate_batch_idx(N, batch_size=batch_size, bar = bar, desc = desc):
            yield dict(
                endog_features = self._preprocess_endog(endog_features[start:end]),
                exog_features = self._preprocess_exog(exog_features[start:end]),
                read_depth = self._preprocess_read_depth(read_depth[start:end]),
            )

    def _get_1cycle_scheduler(self,*, min_learning_rate, max_learning_rate, num_epochs, n_batches_per_epoch, beta):
        
        return pyro.optim.lr_scheduler.PyroLRScheduler(OneCycleLR_Wrapper, 
            {'optimizer' : Adam, 'optim_args' : {'lr' : min_learning_rate, 'betas' : (beta, 0.999)}, 'max_lr' : max_learning_rate, 
            'steps_per_epoch' : n_batches_per_epoch, 'epochs' : num_epochs, 'div_factor' : max_learning_rate/min_learning_rate,
            'cycle_momentum' : False, 'three_phase' : False, 'verbose' : False})

    @wraps_modelfunc(adata_extractor=partial(extract_features, include_features = True), 
        del_kwargs=['features','endog_features','exog_features'])
    def fit(self,*,features, endog_features, exog_features):

        assert(isinstance(self.num_epochs, int) and self.num_epochs > 0)
        assert(isinstance(self.batch_size, int) and self.batch_size > 0)
        assert(isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0)
        if self.max_learning_rate is None:
            self.max_learning_rate = self.min_learning_rate
        else:
            assert(isinstance(self.max_learning_rate, float) and self.max_learning_rate > 0)

        self.features = features
        self.num_endog_features = endog_features.shape[-1]
        self.num_exog_features = exog_features.shape[-1]
        assert(self.num_exog_features == len(self.features))

        self._get_weights()
        
        n_observations = endog_features.shape[0]
        n_batches = self.get_num_batches(n_observations, self.batch_size)

        return self

    @wraps_modelfunc(adata_extractor=extract_features, adata_adder= add_topic_comps,
        del_kwargs=['endog_features','exog_features'])
    def predict(self, endog_features, exog_features, batch_size = 512):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')
        latent_vars = []
        for batch in self._iterate_batches(
                endog_features = endog_features, exog_features = exog_features, 
                batch_size = batch_size, bar = True, training = False, desc = 'Predicting latent vars'):
            latent_vars.append(self.encoder.latent_loc(*batch))

        theta = np.vstack(latent_vars)
        return theta


    @staticmethod
    def centered_boxcox_transform(x, a):
        x = (x**a - 1)/a
        return x - x.mean(-1, keepdims = True)

    @wraps_modelfunc(adata_extractor=extract_features, adata_adder=add_umap_features,
        del_kwargs=['endog_features','exog_features'])
    def get_umap_features(self, endog_features, exog_features, batch_size = 512):
        
        compositions = self.predict(endog_features, exog_features, batch_size=batch_size)
        basis = _gram_schmit_basis(compositions.shape[-1]).T

        return self.centered_boxcox_transform(compositions, 0.25).dot(basis)


    def get_logp(self,*,endog_features, exog_features):
        pass
    
    @wraps_modelfunc(adata_extractor=extract_features, adata_adder= return_output,
        del_kwargs=['endog_features','exog_features'])
    def score(self,*,endog_features, exog_features):
        self.eval()
        return self.get_logp(endog_features = endog_features, exog_features = exog_features)\
            /(endog_features.shape[0] * self.num_exog_features)

    def _batched_impute(self, latent_composition, batch_size = 512, bar = True):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')
        
        for start, end in self._iterate_batch_idx(latent_composition.shape[-1], bar = bar, desc = 'Imputing'):
            yield self.decoder(
                torch.tensor(latent_composition[start : end], requires_grad = False).to(self.device)
            )

    @wraps_modelfunc(adata_extractor=get_obsm, adata_adder=add_imputed_vals,
        del_kwargs=['latent_compositions'])
    def impute(self, latent_composition, batch_size = 512, bar = True):
        return np.vstack(list(self._batched_impute(self, latent_composition, batch_size = batch_size, bar = bar)))

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def _get_save_data(self):
        return dict(
            weights = self.state_dict(),
            params = self.get_params()
        )

    def _set_weights(self, weights):
        self.load_state_dict(weights)
        self.eval()
        self.to_cpu()
        return self

    def _score_features(self):
        score = np.sign(self._get_gamma()) * (self._get_beta() - self._get_bn_mean())/np.sqrt(self._get_bn_var() + self.bn.epsilon)
        return score

    def _get_topics(self):
        return self._score_features()
    
    def _get_beta(self):
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def _get_gamma(self):
        return self.decoder.bn.weight.cpu().detach().numpy()
    
    def _get_bias(self):
        return self.decoder.bn.bias.cpu().detach().numpy()

    def _get_bn_mean(self):
        return self.decoder.bn.running_mean.cpu().detach().numpy()

    def _get_bn_var(self):
        return self.decoder.bn.running_var.cpu().detach().numpy()

    def to_gpu(self):
        self.set_device('cuda:0')
    
    def to_cpu(self):
        self.set_device('cpu')

    def set_device(self, device):
        logger.debug('Moving model to device: {}'.format(device))
        self.device = device
        self = self.to(self.device)