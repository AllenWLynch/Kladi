import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import logging
import tqdm
from kladi.core.plot_utils import map_colors, layout_labels
import re
from sklearn.preprocessing import scale
import warnings
from collections import Counter
from kladi.core.plot_utils import plot_factor_influence

class ClusterISDSplitter:

    def __init__(self, clusters):
        
        assert(isinstance(clusters, (list, np.ndarray)))
        self.clusters = np.ravel(np.array(clusters))
        self.cluster_ids = np.unique(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def _iterate_all_bins(self, bin_size = None):

        for cluster_id in self.cluster_ids:
            yield (None, None, self.clusters == cluster_id)

class CovISD:

    def __init__(self, *, accessibility_model, expression_model, pseudotime_model):
        
        try:
            accessibility_model.factor_hits
            accessibility_model.factor_names
        except AttributeError:
            raise Exception('User must run "get_motif_hits_in_peaks" using accessibility model before running this function.')

        self.accessibility_model = accessibility_model
        self.expression_model = expression_model
        self.pseudotime_model = pseudotime_model

    @staticmethod
    def _get_latent_vars(model, latent_comp = None, matrix = None):

        assert(not latent_comp is None or not matrix is None)

        if latent_comp is None:
            latent_compositions = model.predict(matrix)
        else:
            latent_compositions = latent_comp
            model._check_latent_vars(latent_compositions)

        return latent_compositions.copy()

    def _get_trajectory_latent_vars(self, atac_latent_compositions, expression_latent_compositions, bin_size = 100):

        state_idx, tree_position, bin_masks = list(zip(*self.pseudotime_model._iterate_all_bins(bin_size = bin_size)))

        atac_states, expr_states = [],[]
        for bin_mask in bin_masks:
            atac_states.append(atac_latent_compositions[bin_mask].mean(0, keepdims = True))
            expr_states.append(expression_latent_compositions[bin_mask].mean(0, keepdims = True))

        return np.vstack(atac_states), np.vstack(expr_states), (state_idx, tree_position)


    def _match_TF_hits_to_expr(self):
        
        expression_genes_map = dict(zip(self.expression_model.genes, range(len(self.expression_model.genes))))
        factor_names = np.array(self.accessibility_model.factor_names)
        
        modeled_factor_idx, factor_expression_idx = [],[]
        for idx, factor_name in enumerate(factor_names):
            for parsed_name in self.accessibility_model._parse_motif_name(factor_name):
                if parsed_name in expression_genes_map:
                    modeled_factor_idx.append(idx)
                    factor_expression_idx.append(expression_genes_map[parsed_name])
                    break

        return np.array(modeled_factor_idx), np.array(factor_expression_idx)

    @staticmethod
    def _get_ISD_covariance(isd_tensor, predicted_expression, predicted_tf_expression):

        expr_matrix = predicted_expression.T
        tf_expr = predicted_tf_expression.T

        covariance_matrix = ((isd_tensor - isd_tensor.mean(-1, keepdims = True)) * \
            (expr_matrix - expr_matrix.mean(-1, keepdims = True))[:, np.newaxis, :]).sum(-1) \
            / (expr_matrix.shape[-1] * expr_matrix.std(-1))[:, np.newaxis]

        isd_tf_expr_corr = np.nan_to_num(((isd_tensor - isd_tensor.mean(-1, keepdims = True)) * \
            (tf_expr - tf_expr.mean(-1, keepdims = True))[np.newaxis,:, :]).sum(-1) \
            / (tf_expr.shape[-1] * tf_expr.std(-1)[np.newaxis, :] * isd_tensor.std(-1)))        

        tf_mask = isd_tf_expr_corr > -0.2

        return np.multiply(covariance_matrix, tf_mask)


    def predict(self,*,gene_models, raw_expression = None, expression_latent_compositions = None,
        accessibility_matrix = None, accessibility_latent_compositions = None, bin_size = 100):

        atac_latent_compositions = self._get_latent_vars(self.accessibility_model, accessibility_latent_compositions, accessibility_matrix)
        expression_latent_compositions = self._get_latent_vars(self.expression_model, expression_latent_compositions, raw_expression)

        assert(len(atac_latent_compositions) == len(expression_latent_compositions))

        atac_states, expr_states, state_meta = self._get_trajectory_latent_vars(atac_latent_compositions, expression_latent_compositions, bin_size=bin_size)
        logging.info('Summarized trajectory with {} state changes.'.format(str(len(atac_states))))
        
        #match TF motif names to expression in system
        factor_mask, factor_gene_idx_map = self._match_TF_hits_to_expr()
        motif_hits = self.accessibility_model.factor_hits[factor_mask, : ].copy()
        factor_names = np.array(self.accessibility_model.factor_names)[factor_mask]
        logging.info('Matched {} factors with expression data.'.format(str(len(factor_names))))
        
        #match gene RP models with expression
        expression_genes_map = dict(zip(self.expression_model.genes, range(len(self.expression_model.genes))))
        #if the gene_model's gene is in the scIPM expression model, advance the RP model and record the idx of the gene
        #in the scIPM model
        rp_gene_to_expr_map, gene_models = list(zip(*[(expression_genes_map[gene_model.gene], gene_model) 
            for gene_model in gene_models if gene_model.gene in expression_genes_map]))
        rp_gene_to_expr_map = np.array(rp_gene_to_expr_map)
        logging.info('Matched {} RP models with expression data.'.format(str(len(rp_gene_to_expr_map))))
        #get ISD
        ISD_scores = self.accessibility_model._insilico_deletion(gene_models, motif_hits, atac_states) #genes x TFs x time
        
        predicted_expression = self.expression_model.impute(expr_states)
        
        #remove ISDs if variance is too low (not enough hits around genes)
        successful_isds = ~np.isclose(ISD_scores.var(axis = (0,2)), 0)
        logging.info('Removing {} factors.'.format(str(len(successful_isds) - successful_isds.sum())))

        factor_names = factor_names[successful_isds]
        factor_gene_idx_map = factor_gene_idx_map[successful_isds]
        ISD_scores = ISD_scores[:, successful_isds, :]
        
        #normalize for each TF, then each gene for comparability
        ISD_scores = ISD_scores/np.linalg.norm(ISD_scores, axis=(0,2))[np.newaxis, : , np.newaxis] # normalize across genes and states per each TF
        ISD_scores = ISD_scores - ISD_scores.mean(axis = (1,2))[:, np.newaxis, np.newaxis] # subtract the mean ISD for each gene
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov_score = self._get_ISD_covariance(ISD_scores, 
                predicted_expression[:, rp_gene_to_expr_map], 
                predicted_expression[:, factor_gene_idx_map])
        
        self.gene_names = np.array([model.gene for model in gene_models])
        self.successful_isds = successful_isds
        self.factor_names = factor_names
        self.state_meta = state_meta
        self.tf_gene_cov_score = cov_score
        self.ISD_cube = ISD_scores
        self.state_expr = predicted_expression[:, rp_gene_to_expr_map]
        
        return self

    def rank_factor_influence(self, gene):

        assert(gene in self.gene_names)
        gene_idx = np.argwhere(gene == self.gene_names)[0]

        return self.factor_names[self.tf_gene_cov_score[gene_idx, :].argsort()][0][::-1]
        

    def get_driver_TFs(self, genelist, relationship = 'activating', sort = True):

        assert(relationship in ['activating','supressing'])
        assert(isinstance(genelist, (list, np.ndarray)))

        query_mask = np.isin(self.gene_names, genelist)
        assert(query_mask.sum() > 0)
        logging.info('Matched {} query genes with modeled genes.'.format(str(query_mask.sum())))

        query_scores, background_scores = self.tf_gene_cov_score[query_mask, :], self.tf_gene_cov_score[~query_mask, :]

        results = []
        for factor, q, b in zip(self.factor_names, query_scores.T, background_scores.T):
            stat, pval = mannwhitneyu(q, b, alternative = 'greater' if relationship == 'activating' else 'less')
            results.append((factor, pval, stat))

        if sort:
            return sorted( results, key = lambda x : (x[1],-x[2]) )
        else:
            return results

    def get_TF_gene_interaction_matrix(self):
        return self.tf_gene_cov_score, self.gene_names, self.factor_names
    
    def plot_compare_gene_modules(self, module1, module2, top_n_genes=(None,None), pval_threshold = (1e-3, 1e-3),
        palette = 'coolwarm', ax = None, figsize = (10,7), fontsize = 12, interactive = False, label_closeness = 5, max_label_repeats = 5):
        
        gs1 = self.expression_model.get_top_genes(module1, top_n_genes[0])
        gs2 = self.expression_model.get_top_genes(module2, top_n_genes[1])
        
        _, factor_gene_idx_map = self._match_TF_hits_to_expr()
        factor_gene_idx_map = factor_gene_idx_map[self.successful_isds]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module1_factor_activations = self.expression_model._score_features()[module1][:, np.newaxis][factor_gene_idx_map, :]
            module2_factor_activations = self.expression_model._score_features()[module2][:, np.newaxis][factor_gene_idx_map, :]
        
        hue = np.ravel(module1_factor_activations - module2_factor_activations)
        
        self.plot_compare_genelists(
            gs1, gs2, axlabels = ('-log10 Module {}'.format(str(module1)), '-log10 Module {}'.format(str(module2))),
            pval_threshold = pval_threshold, hue = hue, palette = palette, ax = ax, figsize = figsize, 
            legend_label = 'Relative Expression \n< Module {} - Module {} >'.format(str(module2), str(module1)),
            fontsize = fontsize, interactive = interactive, label_closeness = label_closeness, max_label_repeats = max_label_repeats,
        )
        

    def plot_compare_genelists(self, genelist1, genelist2, axlabels = ('-log10 genelist1','-log10 genelist2'), pval_threshold = (1e-3, 1e-3),
        hue = None, palette = 'coolwarm', hue_order = None, ax = None, figsize = (10,7), legend_label = '', show_legend = True, fontsize = 12,
        interactive = False, color = 'grey', label_closeness = 5, max_label_repeats = 5):
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = figsize)

        factor_names, l1_pvals, _ = list(zip(*self.get_driver_TFs(genelist1, sort=False)))
        _, l2_pvals, _ = list(zip(*self.get_driver_TFs(genelist2, sort=False)))
        
        plot_factor_influence(ax, l1_pvals, l2_pvals, factor_names, pval_threshold = pval_threshold, hue = hue, hue_order = hue_order, 
            palette = palette, legend_label = legend_label, show_legend = show_legend, label_closeness = label_closeness, 
            max_label_repeats = max_label_repeats, axlabels = axlabels, fontsize = fontsize, interactive = False, color = color)