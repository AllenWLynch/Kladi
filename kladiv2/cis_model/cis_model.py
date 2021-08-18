

from functools import partial
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_mean, init_to_value
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
from kladiv2.cis_model.optim import LBFGS as stochastic_LBFGS
import torch.nn.functional as F
import gc
import argparse

logger = logging.getLogger(__name__)

class SaveCallback:

    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, model):
        if model.was_fit:
            model.save(self.prefix)

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
        return [model.gene for model in self.models]

    @property
    def features(self):
        return self.genes

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


    def save(self, prefix):
        for model in self.models:
            model.save(prefix)


    def load(self, prefix):
        for model in self.models:
            model.load(prefix)

    def subset_fit_models(self, was_fit):
        self.models = [model for fit, model in zip(was_fit, self.models) if fit]
        return self

    @wraps_rp_func(lambda self, expr_adata, atac_data, output : self.subset_fit_models(output), bar_desc = 'Fitting models')
    def fit(self, model, features, callback = None):
        try:
            model.fit(features)
        except ValueError:
            logger.warn('{} model failed to fit.'.format(model.gene))
            pass

        if not callback is None:
            callback(model)

        return model.was_fit

    @wraps_rp_func(lambda self, expr_adata, atac_data, output : np.array(output).sum(), bar_desc = 'Scoring')
    def score(self, model, features):
        return model.score(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output: \
        add_imputed_vals(self, expr_adata, np.hstack(output), add_layer = self.model_type + '_prediction'), bar_desc = 'Predicting expression')
    def predict(self, model, features):
        return model.predict(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output: \
        add_imputed_vals(self, expr_adata, np.hstack(output), add_layer = self.model_type + '_logp'), bar_desc = 'Getting logp(Data)')
    def get_logp(self, model, features):
        return model.get_logp(features)

    @wraps_rp_func(lambda self, expr_adata, atac_data, output: \
        add_imputed_vals(self, expr_adata, np.hstack(output), add_layer = self.model_type + '_samples'))
    def _sample_posterior(self, model, features, site = 'prediction'):
        return model.to_numpy(model.get_posterior_sample(features, site))[:, np.newaxis]

    @wraps_rp_func(lambda self, expr_adata, atac_adata, output : output, bar_desc = 'Formatting features')
    def get_features(self, model,features):
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
        self.was_fit = False

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
                d = pyro.sample('logdistance', dist.LogNormal(np.log(10), 1.))

            theta = pyro.sample('theta', dist.Gamma(2., 0.5))
            gamma = pyro.sample('gamma', dist.LogNormal(0., 0.5))
            bias = pyro.sample('bias', dist.Normal(0, 5.))

            if self.use_trans_features:
                with pyro.plate('trans_coefs', trans_features.shape[-1]):
                    a_trans = pyro.sample('a_trans', dist.Normal(0.,1.))

            with pyro.plate('data', len(upstream_weights)):

                f_Z = a[0] * self.RP(upstream_weights, upstream_distances, d[0])\
                    + a[1] * self.RP(downstream_weights, downstream_distances, d[1]) \
                    + a[2] * promoter_weights.sum(-1)

                if self.use_trans_features:
                    f_Z = f_Z + torch.matmul(trans_features, torch.unsqueeze(a_trans, 0).T).reshape(-1)

                prediction = self.bn(f_Z.reshape((-1,1)).float()).reshape(-1)
                pyro.deterministic('unnormalized_prediction', prediction)
                independent_rate = (gamma * prediction + bias).exp()

                rate =  independent_rate/softmax_denom
                pyro.deterministic('prediction', rate)
                mu = read_depth.exp() * rate

                p = mu / (mu + theta)

                pyro.deterministic('prob_success', p)
                NB = dist.NegativeBinomial(total_count = theta, probs = p)
                pyro.sample('obs', NB, obs = gene_expr)


    def _t(self, X):
        return torch.tensor(X, requires_grad=False)

    @staticmethod
    def get_loss_fn():
        return TraceMeanField_ELBO().differentiable_loss

    def get_optimizer(self, params):
        #return torch.optim.LBFGS(params, lr=self.learning_rate, line_search_fn = 'strong_wolfe')
        return stochastic_LBFGS(params, lr = self.learning_rate, history_size = 5,
            line_search = 'Armijo')


    def get_loss_and_grads(self, optimizer, features):
        
        optimizer.zero_grad()

        loss = self.get_loss_fn()(self.model, self.guide, **features)
        loss.backward()

        grads = optimizer._gather_flat_grad()

        return loss, grads

    def armijo_step(self, optimizer, features, update_curvature = True):

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss_fn()(self.model, self.guide, **features)
            return loss

        obj_loss, grad = self.get_loss_and_grads(optimizer, features)

        # compute initial gradient and objective
        p = optimizer.two_loop_recursion(-grad)
        p/=torch.norm(p)
    
        # perform line search step
        options = {'closure': closure, 'current_loss': obj_loss, 'interpolate': True}
        obj_loss, lr, _, _, _, _ = optimizer.step(p, grad, options=options)

        # compute gradient
        obj_loss.backward()
        grad = optimizer._gather_flat_grad()

        # curvature update
        if update_curvature:
            optimizer.curvature_update(grad, eps=0.2, damping=True)

        return obj_loss.detach().item()

    def fit(self, features):

        features = {k : self._t(v) for k, v in features.items()}

        self._get_weights()
        N = len(features['upstream_weights'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with poutine.trace(param_only=True) as param_capture:
                loss = self.get_loss_fn()(self.model, self.guide, **features)

            params = {site["value"].unconstrained() for site in param_capture.trace.nodes.values()}
            optimizer = self.get_optimizer(params)
            early_stopper = EarlyStopping(patience = 3, tolerance = 1e-4)
            update_curvature = False

            self.loss = []
            self.bn.train()
            for i in range(100):

                self.loss.append(
                    float(self.armijo_step(optimizer, features, update_curvature = update_curvature)/N)
                )
                update_curvature = not update_curvature

                if early_stopper(self.loss[-1]):
                    break

        self.was_fit = True
        self.posterior_map = self.guide()

        del optimizer
        del features
        #del self.guide

        return self


    def get_posterior_sample(self, features, site):

        features = {k : self._t(v) for k, v in features.items()}
        self.bn.eval()

        guide = AutoDelta(self.model, init_loc_fn = init_to_value(values = self.posterior_map))
        guide_trace = poutine.trace(guide).get_trace(**features)
        #print(guide_trace)
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace))\
            .get_trace(**features)

        return model_trace.nodes[self.prefix + '/' + site]['value']

    @property
    def prefix(self):
        return self.get_prefix()


    def predict(self, features):
        return self.to_numpy(self.get_posterior_sample(features, 'prediction'))[:, np.newaxis]


    def score(self, features):
        return self.get_logp(features).sum()


    def get_logp(self, features):

        features = {k : self._t(v) for k, v in features.items()}

        p = self.get_posterior_sample(features, 'prob_success')
        theta = self.posterior_map[self.prefix + '/theta']

        logp = dist.NegativeBinomial(total_count = theta, probs = p).log_prob(features['gene_expr'])
        return self.to_numpy(logp)[:, np.newaxis]
        

    def prob_isd(self, features):
        pass

    @staticmethod
    def to_numpy(X):
        return X.clone().detach().cpu().numpy()

    def get_savename(self, prefix):
        return prefix + self.prefix + '.pth'

    def _get_save_data(self):
        return dict(bn = self.bn.state_dict(), guide = self.posterior_map)

    def _load_save_data(self, state):
        self._get_weights()
        self.bn.load_state_dict(state['bn'])
        self.posterior_map = state['guide']

    def save(self, prefix):
        torch.save(self._get_save_data(), self.get_savename(prefix))

    def load(self, prefix):
        state = torch.load(self.get_savename(prefix))
        self._load_save_data(state)


def main(*,atac_adata, rna_adata, expr_model, accessibility_model, genes, save_prefix,
            learning_rate = 1., gene_start_idx = 0, use_trans_features = False):

    from kladiv2.topic_model.expression_model import ExpressionModel
    from kladiv2.topic_model.accessibility_model import AccessibilityModel
    import anndata

    atac_data = anndata.read_h5ad(atac_adata)
    rna_data = anndata.read_h5ad(rna_adata)
    rna_model = ExpressionModel.load(expr_model)
    atac_model = AccessibilityModel.load(accessibility_model)

    with open(genes, 'r') as f:
        genes = [x.strip() for x in f.readlines()]

    cis_model = CisModeler(expr_model=rna_model, accessibility_model=atac_model,
                      learning_rate=learning_rate, use_trans_features=use_trans_features, genes= genes[gene_start_idx:])

    cis_model.fit(expr_adata=rna_data, atac_adata=atac_data, callback = 
                SaveCallback(save_prefix))


'''if __name__ == "__main__":

    parser = argparse.ArgumentParser('Train RP Models')
    parser.add_argument('--rna_adata', type = str)
    parser.add_argument('--atac_adata', type = str)
    parser.add_argument('--atac_model', type = str)
    parser.add_argument('--rna_model', type = str)
    parser.add_argument('--genes', type = str)
    parser.add_argument('--learning_rate', type = float, default=1.)
    parser.add_argument('--gene_start_idx', type = int, default= 0)
    parser.add_argument('--save_prefix', type = str)

    args = parser.parse_args()

    main(
        atac_adata = args.atac_adata, rna_adata = args.rna_adata, 
        expr_model = args.rna_model, accessibility_model = args.atac_model,
        genes = args.genes, learning_rate = args.learning_rate, 
        gene_start_idx = args.gene_start_idx, save_prefix = args.save_prefix
    )
'''


