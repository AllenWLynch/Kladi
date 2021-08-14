

from functools import partial
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_mean
from pyro import poutine
import numpy as np
import torch.distributions.constraints as constraints
import logging
from math import ceil
import time
from pyro.contrib.autoname import scope
from sklearn.base import BaseEstimator
from kladiv2.topic_model.base import EarlyStopping
from kladiv2.core.adata_interface import *
import warnings
from kladiv2.cis_model.optim import SdLBFGS0


class CisModeler:

    def __init__(self,*,
        expr_model, 
        accessibility_model, 
        learning_rate = 1, 
        genes = None,
        use_trans_features = False,
        counts_layer = None):

        self.expr_model = expr_model
        self.accessibility_model = accessibility_model
        self.learning_rate = learning_rate
        self.features = genes
        self.use_trans_features = use_trans_features
        self.counts_layer = counts_layer

        self.models = []
        for gene in genes:
            self.models.append(
                GeneCisModel(
                    gene = gene, 
                    learning_rate = learning_rate, 
                    use_trans_features = use_trans_features
                )
            )


    @property
    def genes(self):
        return self.features

    @property
    def model_type(self):
        if self.use_trans_features:
            return 'trans'
        else:
            return 'cis'

    def _get_masks(self, tss_distance):
        promoter_mask = np.abs(tss_distance) <= 1500
        upstream_mask = np.logical_and(tss_distance < 0, ~promoter_mask)
        downstream_mask = np.logical_and(tss_distance > 0, ~promoter_mask)

        return promoter_mask, upstream_mask, downstream_mask


    def _get_region_weights(self, trans_features, softmax_denom, idx):
        
        model = self.accessibility_model

        def bn(x):
            return (x - model._get_bn_mean()[idx])/np.sqrt(model._get_bn_var()[idx] + model.decoder.bn.eps)

        rate = model._get_gamma()[idx] * bn(trans_features.dot(model._get_beta()[:, idx])) + model._get_bias()[idx]

        region_probabilities = np.exp(rate)/softmax_denom[:, np.newaxis]

        return region_probabilities

    
    def _get_features_for_model(self,*, gene_expr, read_depth, expr_softmax_denom, trans_features, atac_softmax_denom, 
        upstream_idx, downstream_idx, promoter_idx, upstream_distances, downstream_distances):

        features = dict(
            gene_expr = gene_expr,
            read_depth = read_depth,
            softmax_denom = expr_softmax_denom,
            trans_features = trans_features,
            upstream_distances = upstream_distances,
            downstream_distances = downstream_distances,
        )
        for k, idx in zip(['upstream_weights', 'downstream_weights', 'promoter_weights'],
                    [upstream_idx, downstream_idx, promoter_idx]):

            features[k] = self._get_region_weights(trans_features, atac_softmax_denom, idx) * 1e4
        return features


    @wraps_rp_func()
    def fit(self, model, features):
        return model.fit(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output : np.array(output).sum())
    def score(self, model, features):
        return model.score(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output: \
        add_imputed_vals(self, expr_adata, np.hstack(output), add_layer = self.model_type + '_prediction'))
    def predict(self, model, features):
        return model.predict(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output: \
        add_imputed_vals(self, expr_adata, np.hstack(output), add_layer = self.model_type + '_logp'))
    def get_logp(self, model, features):
        return model.get_logp(features)

    @wraps_rp_func(lambda self, expr_adata, atac_adata, output : output)
    def test(self, model,features):
        return features
    

class GeneCisModel:

    def __init__(self,*,
        gene, 
        learning_rate = 1.,
        use_trans_features = False,
    ):
        self.gene = gene
        self.learning_rate = learning_rate
        self.use_trans_features = use_trans_features

    def _get_weights(self):
        pyro.clear_param_store()
        self.bn = torch.nn.BatchNorm1d(1, momentum = 1.0, affine = False)
        self.guide = AutoDelta(self.model, init_loc_fn = init_to_mean)

    def get_prefix(self):
        return ('trans' if self.use_trans_features else 'cis') + '_' + self.gene


    def RP(self, weights, distances, d):
        return (weights * torch.pow(0.5, distances/(1e3 * d))).sum(-1)

    def model(self, 
        gene_expr, 
        softmax_denom,
        read_depth,
        upstream_weights,
        upstream_distances,
        downstream_weights,
        downstream_distances,
        promoter_weights,
        trans_features):

        with scope(prefix = self.get_prefix()):

            with pyro.plate("spans", 3):
                a = pyro.sample("a", dist.HalfNormal(1.))

            with pyro.plate("upstream-downstream", 2):
                d = pyro.sample('logdistance', dist.LogNormal(np.e, 1.))

            theta = pyro.sample('theta', dist.Gamma(2., 0.5))
            gamma = pyro.sample('gamma', dist.LogNormal(0., 0.5))
            bias = pyro.sample('bias', dist.Normal(0, 5.))

            with pyro.plate('data', len(upstream_weights)):

                f_Z = a[0] * self.RP(upstream_weights, upstream_distances, d[0])\
                    + a[1] * self.RP(downstream_weights, downstream_distances, d[1]) \
                    + a[2] * promoter_weights.sum(-1)

                pyro.deterministic('f_Z', f_Z)
                prediction = self.bn(f_Z.reshape((-1,1)).float()).reshape(-1)
        
                independent_rate = (gamma * prediction + bias).exp()

                rate =  independent_rate/softmax_denom
                mu = read_depth.exp() * rate

                p = mu / (mu + theta)

                NB = dist.NegativeBinomial(total_count = theta, probs = p)
                pyro.sample('obs', NB, obs = gene_expr)


    def _t(self, X):
        return torch.tensor(X, requires_grad=False)


    @staticmethod
    def get_loss_fn():
        return TraceMeanField_ELBO().differentiable_loss

    def get_optimizer(self, params):
        return torch.optim.LBFGS(params, lr=self.learning_rate, line_search_fn = 'strong_wolfe')

    def fit(self, features):
        
        features = {k : self._t(v) for k, v in features.items()}

        self._get_weights()
        N = len(features['upstream_weights'])

        loss_fn = self.get_loss_fn()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with poutine.trace(param_only=True) as param_capture:
                loss = loss_fn(self.model, self.guide, **features)

            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}
            optimizer = self.get_optimizer(params)#SdLBFGS0(params, lr = self.learning_rate)#torch.optim.LBFGS(params, lr=self.learning_rate)
            early_stopper = EarlyStopping(patience = 1, tolerance = 1e-3)

            def closure():
                optimizer.zero_grad()
                loss = loss_fn(self.model, self.guide, **features)
                loss.backward()
                return loss

            self.testing_loss = []
            for iternum in range(20):
                self.testing_loss.append(
                    self.to_numpy(optimizer.step(closure))/N
                )

                if early_stopper(self.testing_loss[-1]):
                    break

            return self

    def predict(self, features):
        pass

    def score(self, features):
        pass

    def get_logp(self, features):
        pass

    def prob_isd(self, features):
        pass

    @staticmethod
    def to_numpy(X):
        return X.detach().cpu().numpy()