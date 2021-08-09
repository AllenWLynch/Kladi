

from functools import partial
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm, trange
from pyro.nn import PyroModule
import numpy as np
import torch.distributions.constraints as constraints
import logging
from kladi.matrix_models.ilr import ilr
from math import ceil
import time
from pyro.contrib.autoname import scope

logger = logging.getLogger(__name__)

class EarlyStopping:

    def __init__(self, 
                 tolerance = 1e-4,
                 patience=3,
                 convergence_check = True):
        self.tolerance = tolerance,
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.convergence_check = convergence_check

    def should_stop_training(self, current_loss):
        
        if current_loss is None:
            pass
        else:
            if ((current_loss - self.best_loss) < -self.tolerance[0]) or \
                (self.convergence_check and ((current_loss - self.best_loss) > 10*self.tolerance[0])):
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    return True
                self.wait += 1

            if current_loss < self.best_loss:
                self.best_loss = current_loss

        return False

class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, num_genes, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, num_genes, bias = False)
        self.bn = nn.BatchNorm1d(num_genes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # the output is σ(βθ)
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

class BaseModel(nn.Module):

    I = 50
    #beta = 0.95

    def __init__(self, encoder_model, decoder_model,*,
            num_modules,
            num_exog_features,
            highly_variable,
            hidden,
            num_layers,
            decoder_dropout,
            encoder_dropout,
            use_cuda,
            seed, **encoder_kwargs):
        super().__init__()

        self._set_seeds(seed)
        pyro.clear_param_store()
        torch.cuda.empty_cache()

        assert(isinstance(use_cuda, bool))
        assert(isinstance(num_modules, int) and num_modules > 0)

        self.num_topics = num_modules

        if highly_variable is None:
            highly_variable = np.ones(num_exog_features).astype(bool)
        else:
            assert(isinstance(highly_variable, np.ndarray))
            assert(highly_variable.dtype == bool)
            
            highly_variable = np.ravel(highly_variable)
            assert(len(highly_variable) == num_exog_features)

        self.highly_variable = highly_variable
        self.num_endog_features = int(highly_variable.sum())
        self.num_exog_features = num_exog_features

        self.use_cuda = torch.cuda.is_available() and use_cuda
        if not use_cuda:
            logger.warn('Cuda unavailable. Will not use GPU speedup while training.')

        self.decoder = decoder_model(self.num_exog_features, num_modules, decoder_dropout)
        self.encoder = encoder_model(self.num_endog_features, num_modules, 
            hidden, encoder_dropout, num_layers, **encoder_kwargs)
            
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.max_prob = torch.tensor([0.99999], requires_grad = False)
        self.K = torch.tensor(self.num_topics, requires_grad = False)
        self.set_device(self.device)
        self.training_bar = True

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

    @staticmethod
    def _clr(Z):
        return np.log(Z)/np.log(Z).mean(-1, keepdims = True)

    def _get_batches(self, *args, batch_size = 32, bar = True, desc = None, **kwargs):
        raise NotImplementedError()

    def _check_latent_vars(self, latent_compositions):
        
        assert(isinstance(latent_compositions, np.ndarray))
        assert(len(latent_compositions.shape) == 2)
        assert(latent_compositions.shape[1] == self.num_topics)
        assert(np.isclose(latent_compositions.sum(-1), 1).all())

    def _get_logp(self, X, batch_size = 32):
        
        X = self._validate_data(X)

        batch_loss = []
        for batch in self._get_batches(X, batch_size = batch_size, bar = False):

            loss = float(self.svi.evaluate_loss(*batch))
            batch_loss.append(loss)

        return np.array(batch_loss).sum() #loss is negative log-likelihood

    def _score_features(self):
        score = np.sign(self._get_gamma()) * (self._get_beta() - self._get_bn_mean())/np.sqrt(self._get_bn_var())
        #score = np.sign(self._get_gamma()) * (self._get_beta())
        return score

    def get_topics(self):
        return self._score_features()
    
    def _get_beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def _get_gamma(self):
        return self.decoder.bn.weight.cpu().detach().numpy()
    
    def _get_bias(self):
        return self.decoder.bn.bias.cpu().detach().numpy()

    def _get_bn_mean(self):
        return self.decoder.bn.running_mean.cpu().detach().numpy()

    def _get_bn_var(self):
        return self.decoder.bn.running_var.cpu().detach().numpy()

    def _get_dispersion(self):
        return pyro.param("dispersion").cpu().detach().numpy()

    def to_gpu(self):
        self.set_device('cuda:0')
    
    def to_cpu(self):
        self.set_device('cpu')

    def set_device(self, device):
        logger.debug('Moving model to device: {}'.format(device))
        self.device = device
        self = self.to(self.device)
        self.max_prob = self.max_prob.to(self.device)
        self.K = self.K.to(self.device)

    def _get_latent_MAP(self, *data):
        raise NotImplementedError()

    def predict(self, X, batch_size = 256):

        X = self._validate_data(X)
        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logger.debug('Predicting latent variables ...')
        latent_vars = []
        for i,batch in enumerate(self._get_batches(X, batch_size = batch_size, bar = True, training = False, desc = 'Predicting latent vars')):
            latent_vars.append(self._get_latent_MAP(*batch))

        theta = np.vstack(latent_vars)

        return theta

    def get_UMAP_features(self, X, batch_size = 256):
        return ilr(self.predict(X, batch_size=batch_size))

    def _validate_data(self):
        raise NotImplementedError()

    @staticmethod
    def _warmup_lr_decay(epoch,*,n_batches, lr_decay):
        return min(1., epoch/n_batches)*(lr_decay**int(epoch/n_batches))

    @staticmethod
    def _set_seeds(seed = None):
        if seed is None:
            seed = int(time.time() * 1e7)%(2**32-1)

        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

    def _get_learning_rate_bounds(self, X, num_epochs = 6, eval_every = 10, 
        min_learning_rate = 1e-6, max_learning_rate = 1, batch_size = 32):
    
        X = self._validate_data(X)
        n_batches = self.get_num_batches(X.shape[0], batch_size)

        eval_steps = ceil((n_batches * num_epochs)/eval_every)
        learning_rates = np.exp(np.linspace(np.log(min_learning_rate), np.log(max_learning_rate), eval_steps+1))

        def lr_function(e):
            return learning_rates[e]/learning_rates[0]
        scheduler = pyro.optim.LambdaLR({'optimizer': Adam, 
            'optim_args': {'lr': learning_rates[0], 'betas' : (0.95, 0.999)}, 
            'lr_lambda' : lr_function})

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())
        batches_complete, steps_complete, step_loss = 0,0,0
        learning_rate_losses = []
        
        try:
            for epoch in range(num_epochs):
                #train step
                self.train()
                for batch in self._get_batches(X, batch_size = batch_size):
                    step_loss += float(self.svi.step(*batch))
                    batches_complete+=1
                    
                    if batches_complete % eval_every == 0 and batches_complete > 0:
                        steps_complete+=1
                        scheduler.step()
                        learning_rate_losses.append(step_loss/(eval_every * batch_size * self.num_exog_features))
                        step_loss = 0.0

        except ValueError:
            pass
                    
        return np.array(learning_rates[:len(learning_rate_losses)]), np.array(learning_rate_losses)


    def _estimate_num_epochs(self,*, X, test_X, test_learning_rate, batch_size = 32, max_epochs = 200,
        tolerance = 1e-4, patience = 3):

        X = self._validate_data(X)

        test_lr = test_learning_rate

        optimizer = pyro.optim.Adam(dict(lr = test_lr))
        self.svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())
        self.epoch_loss = []
        early_stopper = EarlyStopping(tolerance=tolerance, patience=patience)
        self.num_epochs_trained = 0

        t = trange(max_epochs, desc = 'Epoch 0', leave = True)
        for epoch in t:
            #train step
            self.train()
            for batch in self._get_batches(X, batch_size = batch_size):
                self.svi.step(*batch)

            self.epoch_loss.append(self.score(test_X))
            self.num_epochs_trained+=1

            t.set_description("Epoch {} done. Recent losses: {}".format(
                        str(epoch + 1),
                        ' --> '.join('{:.3e}'.format(loss) for loss in self.epoch_loss[-5:])
                    ))

            if early_stopper.should_stop_training(self.epoch_loss[-1]):
                logger.info('Stopped training early!')
                break
            
        return self.num_epochs_trained


    @staticmethod
    def _get_decay_rate(a1, a0, periods):
        return np.exp(
            np.log(a1/a0)/periods
        )

    def _get_exp_range_scheduler(self,*,min_learning_rate, max_learning_rate, num_epochs, n_batches_per_epoch, decay = None,
        epochs_per_cycle = None):

        if epochs_per_cycle is None:
            epochs_per_cycle = 4

        if decay is None:
            #decay such that max_lr == min_lr after num_epochs iterations
            decay = self._get_decay_rate(min_learning_rate, max_learning_rate, 2*num_epochs/epochs_per_cycle)

        return pyro.optim.CyclicLR({
                'optimizer' : Adam,
                'optim_args' : {'lr' : min_learning_rate},
                'base_lr' : min_learning_rate, 'max_lr' : max_learning_rate, 'step_size_up' : int(epochs_per_cycle/2 * n_batches_per_epoch), 
                'mode' : 'exp_range', 'gamma' : decay, 'cycle_momentum' : False,
            })

    @staticmethod
    def _warmup_lr_decay(step,*,n_batches_per_epoch, lr_decay):
        return min(1., step/n_batches_per_epoch)*(lr_decay**int(step/n_batches_per_epoch))


    def _get_multiplicative_decay_scheduler(self,*, min_learning_rate, max_learning_rate, num_epochs, n_batches_per_epoch):
        
        decay = self._get_decay_rate(min_learning_rate, max_learning_rate, num_epochs/2)

        lr_function = partial(self._warmup_lr_decay, n_batches_per_epoch = n_batches_per_epoch, lr_decay = decay)
        return pyro.optim.LambdaLR({'optimizer': Adam, 'optim_args': {'lr': max_learning_rate}, 
            'lr_lambda' : lambda e : lr_function(e)})


    def _get_1cycle_scheduler(self,*, min_learning_rate, max_learning_rate, num_epochs, n_batches_per_epoch, beta):
        
        return pyro.optim.lr_scheduler.PyroLRScheduler(OneCycleLR_Wrapper, 
            {'optimizer' : Adam, 'optim_args' : {'lr' : min_learning_rate, 'betas' : (beta, 0.999)}, 'max_lr' : max_learning_rate, 
            'steps_per_epoch' : n_batches_per_epoch, 'epochs' : num_epochs, 'div_factor' : max_learning_rate/min_learning_rate,
            'cycle_momentum' : False, 'three_phase' : False, 'verbose' : False})

    def get_coherence(self, module_num, counts):
        pass
    
    @staticmethod
    def _get_KL_anneal_factor(step_num, *, n_epochs, n_batches_per_epoch, n_cycles):

        total_steps = n_epochs * n_batches_per_epoch
        
        return min(1., (step_num + 1)/(total_steps * 2/3 + 1))

    def fit(self, X,*, min_learning_rate, max_learning_rate, num_epochs = 200, batch_size = 32, beta = 0.95):

        assert(isinstance(num_epochs, int) and num_epochs > 0)
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(isinstance(min_learning_rate, (int, float)) and min_learning_rate > 0)
        if max_learning_rate is None:
            max_learning_rate = min_learning_rate
        else:
            assert(isinstance(max_learning_rate, float) and max_learning_rate > 0)
        
        X = self._validate_data(X)
        n_observations = X.shape[0]
        n_batches = self.get_num_batches(n_observations, batch_size)

        early_stopper = EarlyStopping(tolerance=3, patience=1e-4)

        scheduler = self._get_1cycle_scheduler(min_learning_rate = min_learning_rate, max_learning_rate = max_learning_rate,
            n_batches_per_epoch = n_batches, num_epochs = num_epochs, beta = beta)

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        self.training_loss, self.testing_loss, self.num_epochs_trained = [],[],0
        
        anneal_fn = partial(self._get_KL_anneal_factor, n_epochs = num_epochs, n_batches_per_epoch = n_batches, n_cycles = 3)

        step_count = 0
        self.anneal_factors = []
        try:

            t = trange(num_epochs+1, desc = 'Epoch 0', leave = True) if self.training_bar else range(num_epochs+1)
            _t = iter(t)
            epoch = 0
            while True:
                
                try:
                    next(_t)
                except StopIteration:
                    pass

                self.train()
                running_loss = 0.0
                for batch in self._get_batches(X, batch_size = batch_size):
                    
                    anneal_factor = anneal_fn(step_count)
                    self.anneal_factors.append(anneal_factor)
                    #if batch[0].shape[0] > 1:
                    try:
                        running_loss += float(self.svi.step(*batch, anneal_factor = anneal_factor))
                        step_count+=1
                    except ValueError:
                        raise ModelParamError('Gradient overflow caused parameter values that were too large to evaluate. Try setting a lower learning rate.')

                    if epoch < num_epochs:
                        scheduler.step()
                
                #epoch cleanup
                epoch_loss = running_loss/(X.shape[0] * self.num_exog_features)
                self.training_loss.append(epoch_loss)
                recent_losses = self.training_loss[-5:]

                if self.training_bar:
                    t.set_description("Epoch {} done. Recent losses: {}".format(
                        str(epoch + 1),
                        ' --> '.join('{:.3e}'.format(loss) for loss in recent_losses)
                    ))

                epoch+=1
                if early_stopper.should_stop_training(recent_losses[-1]) and epoch > num_epochs:
                    break

        except KeyboardInterrupt:
            logger.warn('Interrupted training.')

        self.set_device('cpu')
        self.eval()
        return self

    def options_fit(self, X,*, min_learning_rate, max_learning_rate, num_epochs = 200, batch_size = 32, test_X=None,
        policy = '1cycleLR', triangle_decay = None, epochs_per_cycle = None, patience = 3, tolerance = 1e-4):

        assert(isinstance(num_epochs, int) and num_epochs > 0)
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(isinstance(min_learning_rate, float) and min_learning_rate > 0)
        if max_learning_rate is None:
            max_learning_rate = min_learning_rate
        else:
            assert(isinstance(max_learning_rate, float) and max_learning_rate > 0)
        
        X = self._validate_data(X)
        n_observations = X.shape[0]
        n_batches = self.get_num_batches(n_observations, batch_size)

        scheduler = None
        optim_kwargs = dict(
            min_learning_rate = min_learning_rate,
            max_learning_rate = max_learning_rate,
            n_batches_per_epoch = n_batches,
            num_epochs = num_epochs
        )

        early_stopper = EarlyStopping(tolerance=tolerance, patience=patience)

        if policy == 'cyclicLR':
            scheduler = self._get_exp_range_scheduler(**optim_kwargs, epochs_per_cycle = epochs_per_cycle, decay = triangle_decay)
            early_stopper = EarlyStopping(tolerance= tolerance, patience=0)
        elif policy == 'multiplicative_decay':
            scheduler = self._get_multiplicative_decay_scheduler(**optim_kwargs)    
        elif policy == '1cycleLR':
            scheduler = self._get_1cycle_scheduler(**optim_kwargs)
        elif isinstance(policy, pyro.optim):
            logger.info('Using user-provided optimizer')
            scheduler = policy
        else:
            raise Exception('Mode {} is not valid'.format(str(policy)))

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        self.training_loss, self.testing_loss, self.num_epochs_trained = [],[],0
        try:
            if self.training_bar:
                t = trange(num_epochs, desc = 'Epoch 0', leave = True)
            else:
                t = range(num_epochs)

            for epoch in t:
                
                #train step
                self.train()
                running_loss = 0.0
                for batch in self._get_batches(X, batch_size = batch_size):
                    running_loss += self.svi.step(*batch)
                    scheduler.step()
                
                #epoch cleanup
                epoch_loss = running_loss/(X.shape[0] * self.num_exog_features)
                self.training_loss.append(epoch_loss)
                recent_losses = self.training_loss[-5:]

                if not test_X is None:
                    self.testing_loss.append(self.score(test_X))
                    recent_losses = self.testing_loss[-5:]

                if self.training_bar:
                    t.set_description("Epoch {} done. Recent losses: {}".format(
                        str(epoch + 1),
                        ' --> '.join('{:.3e}'.format(loss) for loss in recent_losses)
                    ))

                self.num_epochs_trained+=1

                if policy == 'multiplicative_decay' \
                    or (policy == 'cyclicLR' and epoch % epochs_per_cycle == 0):
                    #or (policy == '1cycleLR' and epoch > (3*num_epochs)//4): don't do early stopping with 1cycleLR, tune #epochs
                    if early_stopper.should_stop_training(recent_losses[-1]):
                        logger.debug('Stopped training early')
                        break

        except KeyboardInterrupt:
            logger.warn('Interrupted training.')

        self.set_device('cpu')
        self.eval()
        return self
                
    def score(self, X):
        self.eval()
        return self._get_logp(X)/(X.shape[0] * self.num_exog_features)

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def _get_save_data(self):
        return dict(weights = self.state_dict())

    def _load_save_data(self, data):
        self.load_state_dict(data['weights'])
        self.eval()
        self.to_cpu()
        return self