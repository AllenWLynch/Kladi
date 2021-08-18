


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
from math import ceil
import time
from pyro.contrib.autoname import scope
from sklearn.base import BaseEstimator
from kladiv2.core.adata_interface import *

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


class Decoder(torch.nn.Module):
    
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


class BaseModel(torch.nn.Module, BaseEstimator):

    I = 50

    @classmethod
    def load_old_model(cls, filename):

        old_model = torch.load(filename)

        params = old_model['params'].copy()
        params['num_topics'] = params.pop('num_modules')
        
        fit_params = {}
        fit_params['num_exog_features'] = len(params['features'])
        fit_params['num_endog_features'] = params['highly_variable'].sum()
        fit_params['highly_variable'] = params.pop('highly_variable')
        fit_params['features'] = params.pop('features')
        
        model = cls(**params)
        model._set_weights(fit_params, old_model['model']['weights'])

        if 'pi' in old_model['model']:
            model.residual_pi = old_model['model']['pi']
        
        return model

    @classmethod
    def load(cls, filename):

        data = torch.load(filename)

        model = cls(**data['params'])
        model._set_weights(data['fit_params'], data['weights'])

        return model

    def __init__(self,
            highly_variable_key = None,
            predict_expression_key = None,
            counts_layer = None,
            num_topics = 16,
            hidden = 128,
            num_layers = 3,
            num_epochs = 40,
            decoder_dropout = 0.2,
            encoder_dropout = 0.1,
            use_cuda = True,
            seed = None,
            min_learning_rate = 1e-6,
            max_learning_rate = 1e-1,
            beta = 0.95,
            batch_size = 64,
            training_bar = True,
            ):
        super().__init__()

        self.highly_variable_key = highly_variable_key
        self.predict_expression_key = predict_expression_key
        self.counts_layer = counts_layer
        self.num_topics = num_topics
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.use_cuda = use_cuda
        self.seed = seed
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.training_bar = training_bar

    def _set_seeds(self):
        if self.seed is None:
            self.seed = int(time.time() * 1e7)%(2**32-1)

        torch.manual_seed(self.seed)
        pyro.set_rng_seed(self.seed)
        np.random.seed(self.seed)

    def _get_weights(self, on_gpu = True):

        pyro.clear_param_store()
        torch.cuda.empty_cache()
        self._set_seeds()

        assert(isinstance(self.use_cuda, bool))
        assert(isinstance(self.num_topics, int) and self.num_topics > 0)
        assert(isinstance(self.features, (list, np.ndarray)))
        assert(len(self.features) == self.num_exog_features)

        use_cuda = torch.cuda.is_available() and self.use_cuda and on_gpu
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
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
            dropout = self.encoder_dropout, 
            num_layers = self.num_layers
        )

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

        _counts_mu, _counts_var = self._get_lognormal_parameters_from_moments(*self._get_gamma_moments(self.I, self.num_topics))
        pseudocount_mu = pyro.param('pseudocount_mu', _counts_mu * torch.ones((self.num_topics,)), constraint = constraints.positive)

        pseudocount_std = pyro.param('pseudocount_std', np.sqrt(_counts_var) * torch.ones((self.num_topics,)), 
                constraint = constraints.positive)

        pyro.module("encoder", self.encoder)

        with pyro.plate("topics", self.num_topics) as k:
            initial_counts = pyro.sample("a", dist.LogNormal(pseudocount_mu, pseudocount_std))


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
        return torch.tensor(X, requires_grad = False).to(self.device)

    def _iterate_batches(self, endog_features, exog_features, batch_size = 32, bar = True, desc = None):
        
        N = endog_features.shape[0]
        read_depth = np.array(exog_features.sum(-1)).reshape((-1,1))

        for start, end in self._iterate_batch_idx(N, batch_size=batch_size, bar = bar, desc = desc):
            yield dict(
                endog_features = self._preprocess_endog(endog_features[start:end], read_depth[start : end]),
                exog_features = self._preprocess_exog(exog_features[start:end]),
                read_depth = self._preprocess_read_depth(read_depth[start:end]),
            )

    def _get_1cycle_scheduler(self, n_batches_per_epoch):
        
        return pyro.optim.lr_scheduler.PyroLRScheduler(OneCycleLR_Wrapper, 
            {'optimizer' : Adam, 'optim_args' : {'lr' : self.min_learning_rate, 'betas' : (self.beta, 0.999)}, 'max_lr' : self.max_learning_rate, 
            'steps_per_epoch' : n_batches_per_epoch, 'epochs' : self.num_epochs, 'div_factor' : self.max_learning_rate/self.min_learning_rate,
            'cycle_momentum' : False, 'three_phase' : False, 'verbose' : False})

    @staticmethod
    def _get_KL_anneal_factor(step_num, *, n_epochs, n_batches_per_epoch):

        total_steps = n_epochs * n_batches_per_epoch
        
        return min(1., (step_num + 1)/(total_steps * 2/3 + 1))

    @property
    def highly_variable(self):
        return self._highly_variable

    @highly_variable.setter
    def highly_variable(self, h):
        assert(isinstance(h, (list, np.ndarray)))
        h = np.ravel(np.array(h))
        assert(h.dtype == bool)
        assert(len(h) == self.num_exog_features)
        self._highly_variable = h

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, f):
        assert(isinstance(f, (list, np.ndarray)))
        f = np.ravel(np.array(f))
        assert(len(f) == self.num_exog_features)
        self._features = f


    def _fit(self,*,features, highly_variable, endog_features, exog_features):

        assert(isinstance(self.num_epochs, int) and self.num_epochs > 0)
        assert(isinstance(self.batch_size, int) and self.batch_size > 0)
        assert(isinstance(self.min_learning_rate, (int, float)) and self.min_learning_rate > 0)
        if self.max_learning_rate is None:
            self.max_learning_rate = self.min_learning_rate
        else:
            assert(isinstance(self.max_learning_rate, float) and self.max_learning_rate > 0)

        self.enrichments = {}
        self.num_endog_features = endog_features.shape[-1]
        self.num_exog_features = exog_features.shape[-1]
        self.features = features
        self.highly_variable = highly_variable
        
        self._get_weights()
        
        n_observations = endog_features.shape[0]
        n_batches = self.get_num_batches(n_observations, self.batch_size)

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4, convergence_check=False)

        scheduler = self._get_1cycle_scheduler(n_batches)
        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        self.training_loss, self.testing_loss, self.num_epochs_trained = [],[],0
        
        anneal_fn = partial(self._get_KL_anneal_factor, n_epochs = self.num_epochs, 
            n_batches_per_epoch = n_batches)

        step_count = 0
        self.anneal_factors = []
        try:

            t = trange(self.num_epochs+1, desc = 'Epoch 0', leave = True) if self.training_bar else range(self.num_epochs+1)
            _t = iter(t)
            epoch = 0
            while True:
                
                try:
                    next(_t)
                except StopIteration:
                    pass

                self.train()
                running_loss = 0.0
                for batch in self._iterate_batches(endog_features = endog_features, exog_features = exog_features, 
                        batch_size = self.batch_size, bar = False):
                    
                    anneal_factor = anneal_fn(step_count)
                    self.anneal_factors.append(anneal_factor)
                    #if batch[0].shape[0] > 1:
                    try:
                        running_loss += float(self.svi.step(**batch, anneal_factor = anneal_factor))
                        step_count+=1
                    except ValueError:
                        raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate. Try setting a lower learning rate.')

                    if epoch < self.num_epochs:
                        scheduler.step()
                
                #epoch cleanup
                epoch_loss = running_loss/(n_observations * self.num_exog_features)
                self.training_loss.append(epoch_loss)
                recent_losses = self.training_loss[-5:]

                if self.training_bar:
                    t.set_description("Epoch {} done. Recent losses: {}".format(
                        str(epoch + 1),
                        ' --> '.join('{:.3e}'.format(loss) for loss in recent_losses)
                    ))

                epoch+=1
                if early_stopper(recent_losses[-1]) and epoch > self.num_epochs:
                    break

        except KeyboardInterrupt:
            logger.warn('Interrupted training.')

        self.set_device('cpu')
        self.eval()
        return self

    @wraps_modelfunc(adata_extractor=fit_adata, 
        del_kwargs=['features','highly_variable','endog_features','exog_features'])
    def fit(self,*,features, highly_variable, endog_features, exog_features):
        return self._fit(features = features, highly_variable = highly_variable, 
            endog_features = endog_features, exog_features = exog_features)


    def _run_encoder_fn(self, fn, batch_size = 512, bar = True, desc = 'Predicting latent vars', **features):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')
        results = []
        for batch in self._iterate_batches(**features,
                batch_size = batch_size, bar = bar,  desc = desc):
            results.append(fn(batch['endog_features'], batch['read_depth']))

        results = np.vstack(results)
        return results


    @wraps_modelfunc(adata_extractor=extract_features, adata_adder= add_topic_comps,
        del_kwargs=['endog_features','exog_features'])
    def predict(self, batch_size = 512, *,endog_features, exog_features):
        return self._run_encoder_fn(self.encoder.topic_comps, batch_size = batch_size, 
            endog_features = endog_features, exog_features = exog_features)


    @staticmethod
    def centered_boxcox_transform(x, a):
        x = (x**a - 1)/a
        return x - x.mean(-1, keepdims = True)

    @staticmethod
    def _gram_schmidt_basis(n):
        basis = np.zeros((n, n-1))
        for j in range(n-1):
            i = j + 1
            e = np.array([(1/i)]*i + [-1] +
                        [0]*(n-i-1))*np.sqrt(i/(i+1))
            basis[:, j] = e
        return basis


    def _project_to_orthospace(self, compositions):

        basis = self._gram_schmidt_basis(compositions.shape[-1])
        return self.centered_boxcox_transform(compositions, 0.25).dot(basis)
    

    @wraps_modelfunc(adata_extractor=extract_features, adata_adder= add_umap_features,
        del_kwargs=['endog_features','exog_features'])
    def get_umap_features(self,batch_size = 512, *,endog_features, exog_features):
        
        compositions = self._run_encoder_fn(self.encoder.topic_comps, endog_features = endog_features, exog_features = exog_features, batch_size=batch_size)
        return self._project_to_orthospace(compositions)


    def _get_logp(self,*,endog_features, exog_features):
        pass
    
    @wraps_modelfunc(adata_extractor=extract_features, adata_adder= return_output,
        del_kwargs=['endog_features','exog_features'])
    def score(self,*,endog_features, exog_features):
        self.eval()
        return self._get_logp(endog_features = endog_features, exog_features = exog_features)\
            /(endog_features.shape[0] * self.num_exog_features)

    
    def _run_decoder_fn(self, fn, latent_composition, batch_size = 512, bar = True, desc = 'Imputing features'):

        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.info('Predicting latent variables ...')

        for start, end in self._iterate_batch_idx(latent_composition.shape[0], batch_size = batch_size, bar = bar, desc = desc):
            yield fn(
                    torch.tensor(latent_composition[start : end], requires_grad = False).to(self.device)
                ).detach().cpu().numpy()


    def _batched_impute(self, latent_composition, batch_size = 512, bar = True):

        return self._run_decoder_fn(self.decoder, latent_composition, 
                    batch_size= batch_size, bar = bar)
        

    @wraps_modelfunc(adata_extractor=get_topic_comps, adata_adder=add_imputed_vals,
        del_kwargs=['topic_compositions'])
    def impute(self, batch_size = 512, bar = True, *, topic_compositions):
        return np.vstack([
            x for x  in self._batched_impute(topic_compositions, batch_size = batch_size, bar = bar)
        ])


    @wraps_modelfunc(adata_extractor = get_topic_comps, adata_adder = partial(add_obs_col, colname = 'softmax_denom'), 
        del_kwargs = ['topic_compositions'])
    def _get_softmax_denom(self, topic_compositions, batch_size = 512, bar = True):
        return np.concatenate([
            x for x in self._run_decoder_fn(self.decoder.get_softmax_denom, topic_compositions, 
                        batch_size = batch_size, bar = bar, desc = 'Calculating softmax summary data')
        ])

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def _get_save_data(self):
        return dict(
            weights = self.state_dict(),
            params = self.get_params(),
            fit_params = dict(
                num_endog_features = self.num_endog_features,
                num_exog_features = self.num_exog_features,
                highly_variable = self.highly_variable,
                features = self.features,
                enrichments = self.enrichments,
            )
        )

    def save(self, filename):
        torch.save(self._get_save_data(), filename)

    def _set_weights(self, fit_params, weights):

        for param, value in fit_params.items():
            setattr(self, param, value)
        
        self._get_weights(on_gpu = False)

        self.load_state_dict(weights)
        self.eval()
        self.to_cpu()
        return self

    def _score_features(self):
        score = np.sign(self._get_gamma()) * (self._get_beta() - self._get_bn_mean())/np.sqrt(self._get_bn_var() + self.decoder.bn.eps)
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
        logger.info('Moving model to device: {}'.format(device))
        self.device = device
        self = self.to(self.device)

    @property
    def topic_cols(self):
        return ['topic_' + str(i) for i in range(self.num_topics)]