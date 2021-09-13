
import anndata
import inspect
from functools import wraps
import numpy as np
import logging
from tqdm import tqdm
from scipy.sparse import isspmatrix
from scipy import sparse

logger = logging.getLogger(__name__)

def wraps_functional(*,
    adata_extractor,
    adata_adder = lambda self, adata, output : None,
    del_kwargs = []
):

    def run(func):

        getter_signature = inspect.signature(adata_extractor).parameters.copy()
        adder_signature = inspect.signature(adata_adder).parameters.copy()
        func_signature = inspect.signature(func).parameters.copy()

        for del_kwarg in del_kwargs:
            func_signature.pop(del_kwarg)

        getter_signature.pop('self')
        adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(adata, **kwargs):

            getter_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(adata_extractor).parameters.copy().keys() if arg in kwargs
            }

            adder_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(adata_adder).parameters.copy().keys() if arg in kwargs
            }

            function_kwargs = {
                arg: kwargs[arg]
                for arg in func_signature.keys() if arg in kwargs
            }

            for kwarg in kwargs.keys():
                if not any(
                    [kwarg in subfunction_kwargs.keys() for subfunction_kwargs in [getter_kwargs, adder_kwargs, function_kwargs]]
                ):
                    raise TypeError('{} is not a valid keyword arg for this function.'.format(kwarg))

            output = func(**adata_extractor(None, adata, **getter_kwargs), **function_kwargs)

            #print(output, adata, adder_kwargs)
            return adata_adder(None, adata, output, **adder_kwargs)

        return _run
    
    return run

def wraps_modelfunc(*,
    adata_extractor,
    adata_adder = lambda self, adata, output : None,
    del_kwargs = []
):

    def run(func):

        getter_signature = inspect.signature(adata_extractor).parameters.copy()
        adder_signature = inspect.signature(adata_adder).parameters.copy()
        func_signature = inspect.signature(func).parameters.copy()

        for del_kwarg in del_kwargs:
            func_signature.pop(del_kwarg)
    
        func_signature.pop('self')
        adder_signature.pop('self')
        adder_signature.pop('adata')
        adder_signature.pop('output')

        getter_signature.update(func_signature)
        getter_signature.update(adder_signature)

        func.__signature__ = inspect.Signature(list(getter_signature.values()))
        
        @wraps(func)
        def _run(self, adata, **kwargs):

            getter_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(adata_extractor).parameters.copy().keys() if arg in kwargs
            }

            adder_kwargs = {
                arg : kwargs[arg]
                for arg in inspect.signature(adata_adder).parameters.copy().keys() if arg in kwargs
            }

            function_kwargs = {
                arg: kwargs[arg]
                for arg in func_signature.keys() if arg in kwargs
            }

            for kwarg in kwargs.keys():
                if not any(
                    [kwarg in subfunction_kwargs.keys() for subfunction_kwargs in [getter_kwargs, adder_kwargs, function_kwargs]]
                ):
                    raise TypeError('{} is not a valid keyword arg for this function.'.format(kwarg))

            output = func(self, **adata_extractor(self, adata, **getter_kwargs), **function_kwargs)

            return adata_adder(self, adata, output, **adder_kwargs)

        return _run
    
    return run

## GENERAL ACCESSORS ##

def fetch_layer(adata, layer):
    if layer is None:
        return adata.X.copy()
    else:
        return adata.layers[layer].copy()

def return_output(self, adata, output):
    return output

def return_adata(self, adata, output):
    return adata

def add_obs_col(self, adata, output,*,colname):
    adata.obs[colname] = output


def project_matrix(adata_index, project_features, vals):

    orig_feature_idx = dict(zip(adata_index, np.arange(len(adata_index))))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    matrix = np.full((vals.shape[0], len(adata_index)), np.nan)
    matrix[:, original_to_imputed_map] = vals

    return matrix


def add_imputed_vals(self, adata, output, add_layer = 'imputed'):
    
    logger.info('Added layer: ' + add_layer)
    adata_features = adata.var_names.values

    orig_feature_idx = dict(zip(adata_features, np.arange(adata.shape[-1])))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in self.features]
    )

    new_layer = np.full(adata.shape, np.nan)
    new_layer[:, original_to_imputed_map] = output

    adata.layers[add_layer] = new_layer


## TOPIC MODEL INTERFACE ##

def fit_adata(self, adata):

    if self.predict_expression_key is None:
        predict_mask = np.ones(adata.shape[-1]).astype(bool)
    else:
        predict_mask = adata.var_vector(self.predict_expression_key)
        logger.info('Predicting expression from genes from col: ' + self.predict_expression_key)
            
    adata = adata[:, predict_mask]

    if self.highly_variable_key is None:
        highly_variable = np.ones(adata.shape[-1].astype(bool))
    else:
        highly_variable = adata.var_vector(self.highly_variable_key)
        logger.info('Using highly-variable genes from col: ' + self.highly_variable_key)

    features = adata.var_names.values

    return dict(
        features = features,
        highly_variable = highly_variable,
        endog_features = fetch_layer(adata[:, highly_variable], self.counts_layer),
        exog_features = fetch_layer(adata, self.counts_layer)
    )

def extract_features(self, adata):

    adata = adata[:, self.features]

    return dict(
        endog_features = fetch_layer(adata[:, self.highly_variable], self.counts_layer),
        exog_features = fetch_layer(adata, self.counts_layer),
    )

def get_topic_comps(self, adata, key = 'X_topic_compositions'):
    logger.info('Fetching key {} from obsm'.format(key))
    return dict(topic_compositions = adata.obsm[key])


def add_topic_comps(self, adata, output, add_key = 'X_topic_compositions', add_cols = True, col_prefix = 'topic_'):

    logger.info('Added key to obsm: ' + add_key)
    adata.obsm[add_key] = output

    if add_cols:
        K = output.shape[-1]
        cols = [col_prefix + str(i) for i in range(K)]
        logger.info('Added cols: ' + ', '.join(cols))
        adata.obs[cols] = output


def add_umap_features(self, adata, output, add_key = 'X_umap_features'):
    logger.info('Added key to obsm: ' + add_key)
    adata.obsm[add_key] = output

## ACCESSIBILITY DATA INTERFACE ##

def get_peaks(self, adata, chrom = 'chr', start = 'start', end = 'end'):

    try:
        return dict(peaks = adata.var[[chrom, start, end]].values.tolist())
    except IndexError:
        raise Exception('Some of columns {}, {}, {} are not in .var'.format(chrom, start, end))


def add_factor_hits_data(self, adata, output,*, factor_type):

    factor_id, factor_name, parsed_name, hits = output

    adata.varm[factor_type + '_hits'] = hits.T.tocsc()
    meta_dict = {
        'id' : list(factor_id),
        'name' : list(factor_name),
        'parsed_name' : list(parsed_name),
        'in_expr_data' : [True]*len(factor_id),
    }
    adata.uns[factor_type] = meta_dict

    logger.info('Added key to varm: ' + factor_type + '_hits')
    logger.info('Added key to uns: ' + factor_type)
    #logger.info('Added key to uns: ' + ', '.join([factor_type + '_' + suffix for suffix in ['id','name','parsed_name','in_expr_data']]))


def add_factor_mask(self, adata, mask,*,factor_type):

    if not factor_type in adata.uns:
        raise KeyError('No metadata for factor type {}. User must run "find_motifs" to add motif data.'.format(factor_type))

    assert(isinstance(mask, (list, np.ndarray)))

    mask = list(mask)
    assert(all([type(x) == bool for x in mask]))
    assert(len(mask) == len(adata.uns[factor_type]['in_expr_data']))

    adata.uns[factor_type]['in_expr_data'] = mask


def get_factor_meta(self, adata, factor_type = 'motifs', mask_factors = True):

    fields = ['id','name','parsed_name']

    try:
        meta_dict = adata.uns[factor_type]
    except KeyError:
        raise KeyError('No metadata for factor type {}. User must run "find_motifs" to add motif data.'.format(factor_type))

    mask = np.array(meta_dict['in_expr_data'])
    col_len = len(mask)

    if not mask_factors:
        mask = np.ones_like(mask).astype(bool)

    metadata = [
        list(np.array(meta_dict[field])[mask])
        for field in fields
    ]

    metadata = list(zip(*metadata))
    metadata = [dict(zip(fields, v)) for v in metadata]

    return metadata, mask


def get_factor_hits(self, adata, factor_type = 'motifs', mask_factors = True, binarize = True):

    try:

        metadata, mask = get_factor_meta(None, adata, factor_type = factor_type, mask_factors = mask_factors)

        hits_matrix = adata[:, self.features].varm[factor_type + '_hits'].T.tocsr()
        hits_matrix = hits_matrix[mask, :]

        if binarize:
            hits_matrix.data = np.ones_like(hits_matrix.data)
    
    except KeyError:
        raise KeyError('User must run "find_motifs" or "find_ChIP_hits" to add binding data before running this function')

    return dict(
        hits_matrix = hits_matrix,
        metadata = metadata
    )


def get_factor_hits_and_latent_comps(self, adata, factor_type = 'motifs', mask_factors = True, key = 'X_topic_compositions'):

    return dict(
        **get_factor_hits(self, adata, factor_type = factor_type, mask_factors = mask_factors),
        **get_topic_comps(self, adata, key = key)
    )


def get_motif_score_adata(self, adata, output):

    metadata, scores, norm_scores = output

    X = anndata.AnnData(
        var = metadata,
        obs = adata.obs.copy(),
        X = norm_scores,
    )
    X.layers['raw_logp_binding'] = scores

    return X


def save_factor_enrichment(self, adata, output,*, module_num, factor_type):

    adata.uns[('enrichment', module_num, factor_type)] = output
    return output

## RP FUNCTION INTERFACE ##

def get_peak_and_tss_data(self, adata, tss_data = None, peak_chrom = 'chr', peak_start = 'start', peak_end = 'end', 
        gene_id = 'geneSymbol', gene_chrom = 'chrom', gene_start = 'txStart', gene_end = 'txEnd', gene_strand = 'strand'):

    if tss_data is None:
        raise Exception('User must provide dataframe of tss data to "tss_data" parameter.')

    return_dict = get_peaks(self, adata, chrom = peak_chrom, start = peak_start, end = peak_end)

    return_dict.update(
        {
            'gene_id' : tss_data[gene_id].values,
            'chrom' : tss_data[gene_chrom].values,
            'start' : tss_data[gene_start].values,
            'end' : tss_data[gene_end].values,
            'strand' : tss_data[gene_strand].values
        }
    )

    return return_dict


def add_peak_gene_distances(self, adata, output):

    distances, genes = output

    adata.varm['distance_to_TSS'] = distances.tocsc()
    adata.uns['distance_to_TSS_genes'] = list(genes)

    logger.info('Added key to var: distance_to_TSS')
    logger.info('Added key to uns: distance_to_TSS_genes')


def wraps_rp_func(adata_adder = lambda self, expr_adata, atac_adata, output, **kwargs : None, 
    bar_desc = '', include_factor_data = False):

    def wrap_fn(func):

        def rp_signature(*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions'):
            pass

        def isd_signature(*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions', factor_type = 'motifs'):
            pass

        func_signature = inspect.signature(func).parameters.copy()
        func_signature.pop('model')
        func_signature.pop('features')
        
        if include_factor_data:
            rp_signature = isd_signature

        mock = inspect.signature(rp_signature).parameters.copy()

        func_signature.update(mock)
        func.__signature__ = inspect.Signature(list(func_signature.values()))
        
        @wraps(func)
        def get_RP_model_features(self,*, expr_adata, atac_adata, atac_topic_comps_key = 'X_topic_compositions', 
            factor_type = 'motifs', **kwargs):

            if not 'model_read_scale' in expr_adata.obs.columns:
                self.expr_model._get_read_depth(expr_adata)

            read_depth = expr_adata.obs_vector('model_read_scale')

            if not 'softmax_denom' in expr_adata.obs.columns:
                self.expr_model._get_softmax_denom(expr_adata)

            expr_softmax_denom = expr_adata.obs_vector('softmax_denom')

            if not 'softmax_denom' in atac_adata.obs.columns:
                self.accessibility_model._get_softmax_denom(atac_adata)

            atac_softmax_denom = atac_adata.obs_vector('softmax_denom')

            if not atac_topic_comps_key in atac_adata.obsm:
                self.accessibility_model.predict(atac_adata, add_key = atac_topic_comps_key, add_cols = False)

            trans_features = atac_adata.obsm[atac_topic_comps_key]

            if not 'distance_to_TSS' in atac_adata.varm:
                raise Exception('Peaks have not been annotated with TSS locations. Run "get_distance_to_TSS" before proceeding.')

            distance_matrix = atac_adata.varm['distance_to_TSS'].T #genes, #regions

            hits_data = dict()
            if include_factor_data:
                hits_data = get_factor_hits(self.accessibility_model, atac_adata, factor_type = factor_type,
                    binarize = True)

            results = []
            for model in tqdm(self.models, desc = bar_desc):

                gene_name = model.gene
                try:
                    gene_idx = np.argwhere(np.array(atac_adata.uns['distance_to_TSS_genes']) == gene_name)[0,0]
                except IndexError:
                    raise IndexError('Gene {} does not appear in peak annotation'.format(gene_name))

                try:
                    gene_expr = fetch_layer(expr_adata[:, gene_name], self.counts_layer)
                    if isspmatrix(gene_expr):
                        gene_expr = gene_expr.toarray()
                    
                    gene_expr = np.ravel(gene_expr)
                except KeyError:
                    raise KeyError('Gene {} is not found in expression data var_names'.format(gene_name))

                peak_idx = distance_matrix[gene_idx, :].indices
                tss_distance = distance_matrix[gene_idx, :].data

                model_features = {}
                for region_name, mask in zip(['promoter','upstream','downstream'], self._get_masks(tss_distance)):
                    model_features[region_name + '_idx'] = peak_idx[mask]
                    model_features[region_name + '_distances'] = np.abs(tss_distance[mask])

                model_features.pop('promoter_distances')
                
                results.append(func(
                        self, model, 
                        self._get_features_for_model(
                            gene_expr = gene_expr,
                            read_depth = read_depth,
                            expr_softmax_denom = expr_softmax_denom,
                            trans_features = trans_features,
                            atac_softmax_denom = atac_softmax_denom,
                            include_factor_data = include_factor_data,
                            **model_features,
                            ),
                        **hits_data,
                        **kwargs)
                )

            return adata_adder(self, expr_adata, atac_adata, results, factor_type = factor_type)

        return get_RP_model_features

    return wrap_fn


def add_isd_results(self, expr_adata, atac_adata, output, factor_type = 'motifs', **kwargs):

    ko_logp, informative_samples = list(zip(*output))

    factor_mask = atac_adata.uns[factor_type]['in_expr_data']
    
    ko_logp = np.vstack(ko_logp).T
    informative_samples = np.vstack(informative_samples).T

    ko_logp = project_matrix(expr_adata.var_names, self.genes, ko_logp)

    projected_ko_logp = np.full((len(factor_mask), ko_logp.shape[-1]), np.nan)
    projected_ko_logp[np.array(factor_mask), :] = ko_logp
    
    expr_adata.varm[factor_type + '-prob_deletion'] = projected_ko_logp.T
    logger.info("Added key to varm: '{}-prob_deletion')".format(factor_type))

    informative_samples = project_matrix(expr_adata.var_names, self.genes, informative_samples)
    informative_samples = np.where(~np.isnan(informative_samples), informative_samples, 0)
    expr_adata.layers[factor_type + '-informative_samples'] = sparse.csr_matrix(informative_samples)
    logger.info('Added key to layers: {}-informative_samples'.format(factor_type))


## PSEUDOTIME STUFF ##

def fetch_diffmap_eigvals(self, adata, diffmap_key = 'X_diffmap'):
    return dict(
        diffmap = adata.obsm[diffmap_key],
        eig_vals = adata.uns['diffmap_evals']
    )


def add_diffmap(self, adata, output, diffmap_key = 'X_diffmap'):
    logging.info('Added key to obsm: {}, normalized diffmap with {} components.'.format(
        diffmap_key,
        str(output.shape[-1])
    ))
    adata.obsm[diffmap_key] = output


def fetch_diffmap_distances(self, adata, diffmap_distances_key = 'X_diffmap'):

    try:
        distance_matrix = adata.obsp[diffmap_distances_key + "_distances"]
        diffmap = adata.obsm[diffmap_distances_key]
    except KeyError:
        raise KeyError(
            '''
You must calculate a diffusion map for the data, and get diffusion-based distances before running this function. Using scanpy:
    
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors = 30, use_rep = "X_diffmap", key_added = "X_diffmap")
            
            '''
        )
    return dict(distance_matrix = distance_matrix, diffmap = diffmap)


def add_transport_map(self, adata, output):

    pseudotime, transport_map, start_cell = output 

    adata.obs['mira_pseudotime'] = pseudotime
    adata.obsp['transport_map'] = transport_map
    adata.uns['iroot'] = start_cell

    logger.info('Added key to obs: mira_pseudotime')
    logger.info('Added key to obsp: transport_map')
    logger.info('Added key to uns: iroot')


def add_branch_probs(self, adata, output):
    adata.obsm['branch_probs'], adata.uns['lineage_names'] = output

    logger.info('Added key to obsm: branch_probs')
    logger.info('Added key to uns: lineage_names')


def fetch_transport_map(self, adata):
    return dict(transport_map = adata.obsp['transport_map'])


def fetch_tree_state_args(self, adata):

    try:
        return dict(
            lineage_names = adata.uns['lineage_names'],
            branch_probs = adata.obsm['branch_probs'],
            pseudotime = adata.obs['mira_pseudotime'].values,
            start_cell = adata.uns['iroot'],
        )
    except KeyError:
        raise KeyError('One of the required pieces to run this function is not present. Make sure you\'ve first run "get_transport_map" and "get_branch_probabilities".')


def add_tree_state_args(self, adata, output):

    adata.obs['tree_states'] = output['tree_states']
    adata.uns['tree_state_names'] = output['state_names']
    adata.uns['connectivities_tree'] = output['tree']

    logger.info('Added key to obs: tree_states')
    logger.info('Added key to uns: tree_state_names')
    logger.info('Added key to uns: connectivities_tree')

## LOCAL/GLOBAL STUFF ##

def fetch_logp_data(self, adata, counts_layer = None):

    try:
        cis_logp = adata.layers['cis_logp']
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained cis_model object before running this function')

    try:
        trans_logp = adata.layers['trans_logp']
    except KeyError:
        raise KeyError('User must run "get_logp" using a trained global_model (set use_trans_features = True on cis_model object) before running this function')

    overlapped_genes = np.logical_and(np.isfinite(cis_logp).all(0), np.isfinite(trans_logp).all(0))
    expression = fetch_layer(adata, counts_layer)

    return dict(
        cis_logp = cis_logp[:, overlapped_genes],
        trans_logp = trans_logp[:, overlapped_genes],
        gene_expr = expression[:, overlapped_genes],
        genes = adata.var_names[overlapped_genes].values,
    )


def project_row(adata_index, project_features, vals, width):

    orig_feature_idx = dict(zip(adata_index, np.arange(width)))

    original_to_imputed_map = np.array(
        [orig_feature_idx[feature] for feature in project_features]
    )

    new_row = np.full(width, np.nan)
    new_row[original_to_imputed_map] = vals
    return new_row
    

def add_global_test_statistic(self, adata, output):

    genes, test_stat, pval, nonzero_counts = output
    
    adata.var['global_regulation_test_statistic'] = \
        project_row(adata.var_names.values, genes, test_stat, adata.shape[-1])

    adata.var['global_regulation_pval'] = \
        project_row(adata.var_names.values, genes, pval, adata.shape[-1])

    adata.var['nonzero_counts'] = \
        project_row(adata.var_names.values, genes, nonzero_counts, adata.shape[-1])

    logger.info('Added keys to var: global_regulation_test_statistic, global_regulation_pval, nonzero_counts')


def fetch_global_test_statistic(self, adata):

    try:
        test_stat = adata.var['global_regulation_test_statistic']
    except KeyError:
        raise KeyError(
            'User must run "global_local_test" function to calculate test_statistic before running this function'
        )

    genes = adata.var_names
    mask = np.isfinite(test_stat)

    return dict(
        genes = genes[mask],
        test_statistic = test_stat[mask],
    )

def fetch_cis_trans_prediction(self, adata):

    try:
        cis = adata.layers['cis_prediction']
    except KeyError:
        raise KeyError('User must run "predict" with a cis_model object before running this function.')

    try:
        trans = adata.layers['trans_prediction']
    except KeyError:
        raise KeyError('User must run "predict" with a cis_model object with "use_trans_features" set to true before running this function.')

    return dict(
        cis_prediction = cis,
        trans_prediction = trans,
    )

def add_chromatin_differential(self, adata, output):
    adata.layers['chromatin_differential'] = output
    logger.info('Added key to layers: chromatin_differential')


def fetch_differential_plot(self, adata, counts_layer = None, genes = None):

    assert(not genes is None)

    if isinstance(genes, str):
        genes = [genes]
    
    adata = adata[:, genes]

    r = fetch_cis_trans_prediction(None, adata)

    try:
        r['chromatin_differential'] = adata.layers['chromatin_differential']
    except KeyError:
        raise KeyError('User must run function "get_cis_differential" before running this function.')

    try:
        r['umap'] = adata.obsm['X_umap']
    except KeyError:
        raise KeyError('X_umap: adata must have a UMAP representation to make this plot.')

    expr = fetch_layer(adata, counts_layer)
    if isspmatrix(expr):
        expr = expr.toarray()

    r['expression'] = expr
    r['gene_names'] = genes

    return r


def fetch_streamplot_data(self, adata, 
        data = None,
        layers = None, 
        pseudotime_key = 'mira_pseudotime',
        group_key = 'tree_states',
        tree_graph_key = 'connectivities_tree', 
        group_names_key = 'tree_state_names',
    ):

    if data is None:
        raise Exception('"data" must be names of columns to plot, not None')
        
    if isinstance(data, str):
        data = [data]
    else:
        assert(isinstance(data, list))

    if isinstance(layers, list):
        assert(len(layers) == len(data))
    else:
        layers = [layers] * len(data)

    if len(set(layers)) > 1 and len(data) > len(set(data)): #multiple layer types, nonunique gene labels
        feature_labels = [
            '{}: {}'.format(col, layer) if not layer is None else col
            for col, layer in zip(data, layers)
        ]
    else:
        feature_labels = data

    tree_graph = None
    try:
        tree_graph = adata.uns[tree_graph_key]
    except KeyError:
        logger.warn(
            'User must run "get_tree_structure" or provide a connectivities matrix of size (groups x groups) to specify tree layout. Plotting without tree structure'
        )

    group_names = None
    try:
        group_names = adata.uns[group_names_key]
    except KeyError:
        logger.warn(
            'No group names provided. Assuming groups are named 0,1,...N'
        )

    pseudotime = adata.obs_vector(pseudotime_key)
    group = adata.obs_vector(group_key)

    columns = [
        adata.obs_vector(col, layer = layer)[:, np.newaxis]
        for col, layer in zip(data, layers)
    ]

    numeric_col = [np.issubdtype(col.dtype, np.number) for col in columns]

    assert(
        all(numeric_col) or not any(numeric_col)
    ), 'All plotting features must be either numeric or nonnumeric. Cannot mix.'

    features = np.hstack(columns)

    return dict(
        pseudotime = pseudotime,
        tree_graph = tree_graph,
        group_names = group_names,
        group = group,
        features = features,
        feature_labels = feature_labels,
    )