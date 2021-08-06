from kladi.matrix_models.scipm_base import BaseModel


def _parse_adata(self, adata, subset = None):

    assert(self._impute_expression is None or self._impute_expression in adata.var.columns)
    assert(self._highly_variable is None or self._highly_variable in adata.var.columns)
    assert(self._train_set is None or self._train_set in adata.obs.columns)
    assert(layer is None or layer in adata.layers)

    exog_features = adata[:, self.get_column_mask(adata.var, self._highly_variable)]

    endog_features = adata[:, self._get_column_mask(adata.var, self._impute_expression)]



class ExpressionModel(BaseModel):

    def __init__(self,
        highly_variable = 'highly_variable',
        impute_expression = None,
        layer = None,
        train_set = None):

    
        self._highly_variable = highly_variable
        self._imputed_expression = impute_expression
        self._train_set = train_set
        self._layer  layer

        highly_variable_is_subset = np.logical_and(
                self._get_column_mask(adata.var, highly_variable),
                self._get_column_mask(adata.var, imputed_expression)
            ).sum() == highly_variable.sum()
        
        if not highly_variable_is_subset:
            logging.warn('Some input / highly variable genes are not predicted by the decoder, so they will not affect reconstruction loss. If this is unintentional, make sure that all highly variable genes are also being imputed.')

        self._features = adata.var_names[self._get_column_mask(adata.var, self._highly_variable)]
        

    def _get_column_mask(self, df, col):
        if col is None:
            return np.ones(len(df)).astype(bool)
        else:
            mask = df[col].values
            assert (mask.dtype == bool)

            return mask


