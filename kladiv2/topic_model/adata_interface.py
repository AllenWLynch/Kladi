
import anndata
import inspect
from functools import wraps
import numpy as np


def fetch_layer(adata, layer):
    if layer is None:
        return adata.X.copy()
    else:
        return adata.layers[layer].copy()
            

def extract_features(self, adata, include_features = False):

    if self.predict_expression is None:
        predict_mask = np.ones(adata.shape[-1]).astype(bool)
    else:
        predict_mask = adata.var_vector(self.predict_expression)

    if self.highly_variable is None:
        highly_variable = np.ones(adata.shape[-1].astype(bool))
    else:
        highly_variable = adata.var_vector(self.highly_variable)

    assert(predict_mask[highly_variable].all()), 'Highly-varible features must be a subset of predicted features.'

    try:
        features = self.features
    except AttributeError:
        features = adata[:, predict_mask].var_names

    shuffled = adata[:, features]

    data =  dict(
        endog_features = fetch_layer(shuffled[:, highly_variable], self.counts_layer),
        exog_features = fetch_layer(shuffled[:, predict_mask], self.counts_layer),
    )

    if include_features:
        data['features'] = features

    return data

def add_topic_comps(adata, X, key = 'X_topic_compositions'):
    adata.obsm[key] = X

def add_umap_features(adata, X, key = 'X_umap_features'):
    adata.obsm[key] = X

def return_output(adata, X):
    return X

def return_adata(adata, X):
    return adata

def add_imputed_vals(adata, X, layer = 'imputed'):
    pass

def get_obsm(self, adata,*,key):
    return adata.obsm[key]

def wraps_modelfunc(*,
    adata_extractor,
    adata_adder = lambda adata, ouput, **kwargs : None,
    del_kwargs = []
):

    def run(func):

        getter_signature = inspect.signature(adata_extractor).parameters.copy()
        adder_signature = inspect.signature(adata_adder).parameters.copy()
        func_signature = inspect.signature(func).parameters.copy()

        for param in list(func_signature.keys()):
            if param in del_kwargs:
                func_signature.pop(param)
        func_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('X')

        func_signature.update(getter_signature)
        func_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(func_signature.values()))
        
        @wraps(func)
        def _run(self, adata, **kwargs):

            getter_kwargs = {
                arg : kwargs.pop(arg) for arg in getter_signature.keys() if arg in kwargs
            }

            adder_kwargs = {
                arg : kwargs.pop(arg) for arg in adder_signature.keys() if arg in kwargs
            }

            output = func(self, **adata_extractor(self, adata, **getter_kwargs), **kwargs)

            return adata_adder(adata, output, **adder_kwargs)

        return _run
    
    return run