

from functools import partial
from kladi.matrix_models.estimator import ModuleEstimator
import os
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm, trange
from pyro.nn import PyroModule
import numpy as np
import torch.distributions.constraints as constraints
import logging
from kladi.matrix_models.ilr import ilr
import configparser
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit


class EarlyStopping:

    def __init__(self, 
                 tolerance = 1e-4,
                 patience=3):
        self.tolerance = tolerance,
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15

    def should_stop_training(self, current_loss):
        
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.tolerance[0]:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    return True
                self.wait += 1

        return False

class BaseModel(nn.Module):

    @classmethod
    def bayesian_tune(cls,*, X, features, highly_variable=None,
        batch_size = 32, num_epochs =200, use_cuda = True, n_fits_per_model = 3, test_proportion = 0.2, num_fits = 50, verbose = 3):
        ''' 
        Use Bayesian Optimization search parameter space for most optimal hyperparameter combinations. Will take fewer steps to find optimal model than a comprehensive sweep using GridSearchCV,
        and may find more optimal model in same amount of time compared to RandomSearchCV.

        Args:

        Args marked with * are optimized by random search

            Model structure:
                X (np.darrary, scipy.sparsematrix): count matrix, raw GEX counts or peak-count matrix
                features (np.darray, list): feature names,
                    for scRNA-seq: list of gene symbols corresponding to columns of count matrix
                    for scATAC-seq: list of format [(chr, start, end), ...] for each peak/column of peak-count matrix. Alternatively, user may pass list of ['chr:start-end', ...] for each peak.
                highly_variable (np.ndarray): for expression model only, boolean mask of same length as ``genes``. Genes flagged with ``True`` will be used as features for encoder. All genes will be used as features for decoder.
                    This allows one to impute many genes while only learning modules on highly variable genes, decreasing model complexity and training time.
            
            Training:
                batch_size (int): batch size for model
                num_epochs (int): maximum number of epochs for training
                use_cuda (bool): use CUDA to accelerate training on GPU (if GPU is available)

            RandomSearch:
                n_fits_per_model (int): number of partitions of dataset for cross-validation
                test_proportion (float between 0,1): proportion of dataset to partition into test set for each fit
                num_fits: number of hyperparameter combinations to try
                verbose: verbosity of RandomSearchCV

        Returns:
            (ExpressionModel): Best model trained from random search
            (skopt.BayesianCV): CV object

        '''
        raise NotImplementedError()


    @classmethod
    def random_tune(cls,*, X, features,highly_variable = None,
        num_modules = [5,10,15,20,25,30,35], dropout = [0.2,0.25,0.3,0.4], intitial_counts = [10,15,20,30,50], 
        hidden = [64,128,256], learning_rate = [0.1,0.005,0.01], lr_decay = [0.95,0.9,0.85,0.8], tolerance = [1e-4,1e-3], 
        batch_size = 32, num_epochs = 200, use_cuda = True,
        n_fits_per_model = 3, test_proportion = 0.2, num_fits = 50, verbose = 3):
        ''' 
        Use random search to search space for best hyperparameters for dataset. Returns best model trained and sklearn CV object with training results.
        Some parameters passed to this method may be optimized, others are fixed by user.

        Args:

        Args marked with * are optimized by random search

            Model structure:
                X (np.darrary, scipy.sparsematrix): count matrix, raw GEX counts or peak-count matrix
                features (np.darray, list): feature names,
                    for scRNA-seq: list of gene symbols corresponding to columns of count matrix
                    for scATAC-seq: list of format [(chr, start, end), ...] for each peak/column of peak-count matrix. Alternatively, user may pass list of ['chr:start-end', ...] for each peak.
                highly_variable (np.ndarray): for expression model only, boolean mask of same length as ``genes``. Genes flagged with ``True`` will be used as features for encoder. All genes will be used as features for decoder.
                    This allows one to impute many genes while only learning modules on highly variable genes, decreasing model complexity and training time.
                num_modules* (list of int): list of number of gene modules
                initial_counts* (list of int): list of sparsity parameters, related to pseudocounts of dirichlet prior. Increasing will lead to denser cell latent variables, decreasing will lead to more sparse latent variables.
                dropout* (list of float between 0,1): list of dropout rates for model
                hidden* (list of int): list of number of nodes in encoder hidden layers

            Training:
                learning_rate* (list of float): list of model learning rates
                lr_decay* (list of float): list per epoch decay of learning rate
                tolerance* (list of float): list of stop criterion for model training
                batch_size (int): batch size for model
                num_epochs (int): maximum number of epochs for training
                use_cuda (bool): use CUDA to accelerate training on GPU (if GPU is available)

            RandomSearch:
                n_fits_per_model (int): number of partitions of dataset for cross-validation
                test_proportion (float between 0,1): proportion of dataset to partition into test set for each fit
                num_fits: number of hyperparameter combinations to try
                verbose: verbosity of RandomSearchCV

        Returns:
            (ExpressionModel): Best model trained from random search
            (sklearn.model_selection.RandomSearchCV): CV object

        '''

        params = dict(
            num_module = num_modules,
            dropout = dropout,
            initial_counts = intitial_counts,
            hidden = hidden,
            learning_rate = learning_rate,
            lr_decay = lr_decay,
            tolerance = tolerance
        )

        estimator = ModuleEstimator(base_estimator = cls, highly_variable = highly_variable,
            features = features, num_epochs = num_epochs, batch_size = batch_size, use_cuda = use_cuda)

        cv = RandomizedSearchCV(
                estimator,
                params,
                verbose = verbose,
                num_iter = num_fits,
                cv = ShuffleSplit(n_splits = n_fits_per_model, test_size = test_proportion),
                refit = True,
        ).fit(X)

        return cv.best_estimator_.estimator, cv


    def __init__(self, num_features, encoder_model, decoder_model, num_topics=15, initial_counts = 15, dropout = 0.2, hidden = 128, use_cuda = True):
        super().__init__()
        assert(isinstance(initial_counts, int) and initial_counts > 0)
        assert(isinstance(dropout, float) and dropout > 0 and dropout < 1)
        assert(isinstance(hidden, int) and hidden > 0)
        assert(isinstance(use_cuda, bool))
        assert(isinstance(num_topics, int) and num_topics > 0)
        assert(isinstance(num_features, int) and num_features > 0)

        a = initial_counts/num_topics
        self.prior_mu = 0
        self.prior_std = np.sqrt(1/a * (1-2/num_topics) + 1/(num_topics * a))

        self.num_features = num_features
        self.num_topics = num_topics
        self.decoder = decoder_model
        self.encoder = encoder_model

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logging.info('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.to(self.device)
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)
        self.training_bar = True


    @staticmethod
    def get_num_batches(N, batch_size):
        return N//batch_size + int(N % batch_size > 0)

    @staticmethod
    def _iterate_batch_idx(N, batch_size, bar = False):

        num_batches = N//batch_size + int(N % batch_size > 0)
        for i in range(num_batches) if not bar else tqdm(range(num_batches)):
            yield i * batch_size, (i + 1) * batch_size

    @staticmethod
    def _clr(Z):
        return np.log(Z)/np.log(Z).mean(-1, keepdims = True)


    def _get_batches(self, *args, batch_size = 32, bar = True):
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

            loss = self.svi.evaluate_loss(*batch)
            batch_loss.append(loss)

        return np.array(batch_loss).sum() #loss is negative log-likelihood

    def _numpy_forward(self, latent_comp):

        activation = (np.dot(latent_comp, self._get_beta()) - self._get_bn_mean())/self._get_bn_var()
        logit = self._get_gamma() * activation  + self._get_bias()

        return np.exp(logit)/np.exp(logit).sum(-1, keepdims = True)
    
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
        logging.info('Moving model to device: {}'.format(device))
        self.device = device
        self = self.to(self.device)
        self.max_prob = self.max_prob.to(self.device)


    def _get_latent_MAP(self, *data):
        raise NotImplementedError()


    def predict(self, X, batch_size = 256):

        X = self._validate_data(X)
        assert(isinstance(batch_size, int) and batch_size > 0)
        
        self.eval()
        logging.info('Predicting latent variables ...')
        latent_vars = []
        for i,batch in enumerate(self._get_batches(X, batch_size = batch_size, training = False)):
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

    def fit(self, X,*, num_epochs = 200, batch_size = 32, test_proportion = 0.2, learning_rate = 0.1, tolerance = 1e-4,
        lr_decay = 0.9, patience = 4):

        eval_every = 1
        assert(isinstance(num_epochs, int) and num_epochs > 0)
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(isinstance(learning_rate, float) and learning_rate > 0)
        assert(isinstance(eval_every, int) and eval_every > 0)
        assert(isinstance(test_proportion, float) and test_proportion >= 0 and test_proportion < 1)

        seed = 2556
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        np.random.seed(seed)

        pyro.clear_param_store()
        logging.info('Validating data ...')

        X = self._validate_data(X)
        n_observations = X.shape[0]
        n_batches = self.get_num_batches(n_observations, batch_size)

        logging.info('Initializing model ...')

        #scheduler = pyro.optim.ExponentialLR({'optimizer': Adam, 'optim_args': {'lr': learning_rate}, 'gamma': lr_decay})
        lr_function = partial(self._warmup_lr_decay, n_batches = n_batches, lr_decay = lr_decay)
        scheduler = pyro.optim.LambdaLR({'optimizer': Adam, 'optim_args': {'lr': learning_rate}, 
            'lr_lambda' : lambda e : lr_function(e)})

        self.svi = SVI(self.model, self.guide, scheduler, loss=TraceMeanField_ELBO())

        self.training_loss, self.learning_rate = [], []
        early_stopper = EarlyStopping(tolerance= tolerance, patience=patience)

        try:
            if self.training_bar:
                t = trange(num_epochs, desc = 'Epoch 0', leave = True)
            else:
                t = range(num_epochs)

            batches_complete = 0
            for epoch in t:
                
                #train step
                self.train()
                running_loss = 0.0
                for batch in self._get_batches(X, batch_size = batch_size):
                    loss = self.svi.step(*batch)
                    running_loss += loss
                    batches_complete+=1
                    scheduler.step()
                    self.learning_rate.append(lr_function(batches_complete))
                
                #epoch cleanup
                epoch_loss = running_loss/(X.shape[0] * self.num_exog_features)
                self.training_loss.append(epoch_loss)

                if self.training_bar:
                    t.set_description("Epoch {} done. Recent losses: {}".format(
                        str(epoch + 1),
                        ' --> '.join('{:.3e}'.format(loss) for loss in self.training_loss[-5:])
                    ))

                if early_stopper.should_stop_training(epoch_loss):
                    logging.info('Stopped training early!')
                    break

        except KeyboardInterrupt:
            logging.warn('Interrupted training.')

        self.set_device('cpu')
        self.eval()
        return self

    def score(self, X):
        self.eval()
        return -self._get_logp(X)/(X.shape[0] * self.num_exog_features)

    def _to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()