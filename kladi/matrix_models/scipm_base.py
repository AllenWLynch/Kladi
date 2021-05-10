

import os
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import tqdm
from pyro.nn import PyroModule
import numpy as np
import torch.distributions.constraints as constraints
import fire
from pyro.infer import Predictive
import logging
import pickle

logging.basicConfig(level = logging.INFO)
logger = logging.Logger('ExprLDA')
logger.setLevel(logging.INFO)

class BaseModel(nn.Module):

    encoder_features_index = 1

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
        self.decoder = decoder_model(num_features, num_topics, dropout)
        self.encoder = encoder_model(num_features, num_topics, hidden, dropout)

        self.use_cuda = torch.cuda.is_available() and use_cuda
        logging.info('Using CUDA: {}'.format(str(self.use_cuda)))
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.to(self.device)
        self.max_prob = torch.tensor([0.99999], requires_grad = False).to(self.device)

    @staticmethod
    def iterate_batch_idx(N, batch_size, bar = False):

        num_batches = N//batch_size + int(N % batch_size > 0)
        for i in range(num_batches) if not bar else tqdm(range(num_batches)):
            yield i * batch_size, (i + 1) * batch_size

    @staticmethod
    def clr(Z):
        return np.log(Z)/np.log(Z).mean(-1, keepdims = True)


    def get_batches(self, *args, batch_size = 32, bar = True):
        raise NotImplementedError()

    def get_logp(self, X, batch_size = 32):
        
        X = self.validate_data(X)

        batch_loss = []
        for batch in self.get_batches(X, batch_size = batch_size, bar = False):

            loss = self.svi.evaluate_loss(*batch)
            batch_loss.append(loss)

        return np.array(batch_loss).sum() #loss is negative log-likelihood

    
    def get_beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T.numpy()

    def get_gamma(self):
        return self.decoder.bn.weight.cpu().detach().numpy()
    
    def get_bias(self):
        return self.decoder.bn.bias.cpu().detach().numpy()

    def get_bn_mean(self):
        return self.decoder.bn.running_mean.cpu().detach().numpy()

    def get_bn_var(self):
        return self.decoder.bn.running_var.cpu().detach().numpy()

    def get_dispersion(self):
        return pyro.param("dispersion").cpu().detach().numpy()


    def to_gpu(self):
        self.set_device('cuda:0')
    
    def to_cpu(self):
        self.set_device('cpu')

    def set_device(self, device):
        logging.info('Moving model to device: {}'.format(device))
        self.device = device
        self.to(self.device)


    def get_latent_MAP(self, *data):
        raise NotImplementedError()


    def predict(self, X, batch_size = 32):

        X = self.validate_data(X)
        assert(isinstance(batch_size, int) and batch_size > 0)

        latent_vars = []
        for i,batch in enumerate(self.get_batches(X, batch_size = batch_size)):
            latent_vars.append(self.get_latent_MAP(*batch))

        theta = np.vstack(latent_vars)

        return theta

    def validate_data(self):
        raise NotImplementedError()


    def fit(self, X, num_epochs = 100, 
            batch_size = 32, learning_rate = 1e-3, eval_every = 1, test_proportion = 0.05, use_validation_set = False):

        assert(isinstance(num_epochs, int) and num_epochs > 0)
        assert(isinstance(batch_size, int) and batch_size > 0)
        assert(isinstance(learning_rate, float) and learning_rate > 0)
        assert(isinstance(eval_every, int) and eval_every > 0)
        assert(isinstance(test_proportion, float) and test_proportion >= 0 and test_proportion < 1)
        assert(isinstance(use_validation_set, bool))

        seed = 2556
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        # set numpy random seed

        pyro.clear_param_store()
        logging.info('Validating data ...')

        X = self.validate_data(X)
        n_observations = X.shape[0]

        logging.info('Initializing model ...')

        adam_params = {"lr": 1e-3}
        optimizer = Adam(adam_params)
        self.svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())

        if use_validation_set:
            logging.warn('Using hold-out set of cells for validation loss. When computing your final model, set "use_validation_set" to False so that all cells are used in computaiton of latent variables. This ensures that holdout-set cells\' latent variables do not have different quality/distribution from train-set cells')
            test_set = np.random.rand(n_observations) < test_proportion
            logging.info("Training with {} cells, testing with {}.".format(str((~test_set).sum()), str(test_set.sum())))
        else:
            test_set = np.zeros(n_observations).astype(np.bool)
        train_set = ~test_set

        logging.info('Training ...')
        self.training_loss = []
        self.testing_loss = []

        self.train()
        try:
            for epoch in range(1, num_epochs + 1):
                running_loss = 0.0
                for batch in self.get_batches(X[train_set], batch_size = batch_size):
                    loss = self.svi.step(*batch)
                    running_loss += loss
                
                epoch_loss = running_loss/train_set.sum()
                self.training_loss.append(epoch_loss)
                logging.info('Epoch {}/{} complete. Training loss: {:.3e}'.format(str(epoch), str(num_epochs), epoch_loss))

                if (epoch % eval_every == 0 or epoch == num_epochs) and test_set.sum() > 0:
                    self.eval()
                    test_loss = self.get_logp(X[test_set])/test_set.sum()
                    logging.info('Test loss: {:.4e}'.format(test_loss))
                    self.train()
                    self.testing_loss.append(test_loss)

        except KeyboardInterrupt:
            logging.error('Interrupted training.')

        self.eval()
        self.set_device('cpu')
        return self

    def to_tensor(self, val):
        return torch.tensor(val).to(self.device)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()