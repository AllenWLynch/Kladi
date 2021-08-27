
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
from kladiv2.plots.base import map_colors
from matplotlib.patches import Patch
import warnings
from scipy.signal import savgol_filter
from sklearn.preprocessing import minmax_scale
import networkx as nx
from functools import partial
from kladiv2.tools.pseudotime import get_dendogram_levels, get_root_state, is_leaf
from kladiv2.plots.base import map_plot
import kladiv2.core.adata_interface as adi
from kladiv2.plots.swarmplot import _plot_swarm_segment, _get_swarm_colors


def _plot_fill(is_top = False,*, ax, time, fill_top, fill_bottom, color, linecolor, linewidth, alpha = 1.):

    ax.fill_between(time, fill_top, fill_bottom, color = color, alpha = alpha)

    if not linecolor is None:
        ax.plot(time, fill_top, color = linecolor, linewidth = linewidth)
        if is_top:
            ax.plot(time, fill_bottom, color = linecolor, linewidth = linewidth)


def _plot_stream_segment(is_leaf = True, centerline = 0, window_size = 101, hide_feature_threshold = 0.05, center_baseline = True, is_root = True,
        palette = 'Set3', linecolor = 'lightgrey', linewidth = 0.1, feature_labels = None, hue_order = None, show_legend = True, max_bar_height = 0.5,
        color = 'black', min_pseudotime = 0., size=20, lineage_name = '', segment_connection = None,*, ax, features, pseudotime, **kwargs,):

    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_features = features.shape[-1]
    if not feature_labels is None:
        assert(len(feature_labels) == num_features)
    else:
        feature_labels = np.arange(num_features).astype(str)

    features = features[np.argsort(pseudotime)] #sort by time
    features = savgol_filter(features, window_size, 1, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    features = np.where(features < hide_feature_threshold, 0., features)
    features = np.cumsum(features, axis=-1)
    
    if center_baseline:
        baseline_adjustment = (features[:,-1]/2)[:, np.newaxis]
    else:
        baseline_adjustment = np.zeros((features.shape[0], 1))

    feature_fill_positions = features - baseline_adjustment + centerline
    feature_bottom = centerline - baseline_adjustment.reshape(-1)
    linecolor = linecolor if num_features > 1 else color
    
    plot_kwargs = dict(
        ax = ax, time = ascending_time,
        linecolor = linecolor, linewidth = linewidth
    )

    legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5))
    
    if num_features == 1:
        _plot_fill(is_top=True, fill_top= feature_fill_positions[:,0], fill_bottom = feature_bottom, color = color, **plot_kwargs)
    else:
        for i, color in enumerate(
                map_colors(ax, feature_labels[::-1], add_legend = is_root and show_legend, hue_order = hue_order[::-1] if not hue_order is None else None, legend_kwargs = legend_params,
                          palette = palette)
            ):
            _plot_fill(is_top = i == 0, fill_top= feature_fill_positions[:, num_features - 1 - i], fill_bottom = feature_bottom, 
                color = color, **plot_kwargs)

    #box in borderlines
    ax.vlines(ascending_time[0], ymin = feature_bottom[0], ymax = feature_fill_positions[0, -1], 
        color = linecolor, linewidth = linewidth)

    ax.vlines(ascending_time[-1], ymin = feature_bottom[-1], ymax = feature_fill_positions[-1, -1], 
        color = linecolor, linewidth = linewidth)


def _plot_scatter_segment(is_leaf = True, centerline = 0, window_size = 101, is_root = True,
        palette = 'Set3', linecolor = 'lightgrey', linewidth = 0.1, feature_labels = None, hue_order = None, show_legend = True, size = 2,
        color = 'black', min_pseudotime = 0., max_bar_height = 0.5,*, ax, features, pseudotime, **kwargs,):
    
    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_features = features.shape[-1]
    if not feature_labels is None:
        assert(len(feature_labels) == num_features)
    else:
        feature_labels = np.arange(num_features).astype(str)

    features = features[np.argsort(pseudotime)] #sort by time
    smoothed_features = savgol_filter(features, window_size, 1, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5))

    min_time, max_time = pseudotime.min(), pseudotime.max()
    centerline -= max_bar_height/2
    
    def scatter(features, smoothed_features, color):

        ax.scatter(
            ascending_time,
            centerline + features,
            color = color,
            s = size
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(xticks = [], yticks = [])
        
        ax.plot(
            ascending_time,
            smoothed_features + centerline,
            color = color, 
            alpha = 0.5,
        )
        
        ax.vlines(min_time, ymin = centerline, ymax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
        ax.vlines(max_time, ymin = centerline, ymax = centerline + max_bar_height, color = linecolor, linewidth = linewidth)
        #ax.hlines(centerline, xmin = min_time, xmax = max_time, color = linecolor, linewidth = linewidth)
    
    if num_features == 1:
        scatter(features[:,0], smoothed_features[:,0], color)
        
    else:
        for i, color in enumerate(
                map_colors(ax, feature_labels[::-1], add_legend = is_root and show_legend, hue_order = hue_order[::-1] if not hue_order is None else None, legend_kwargs = legend_params,
                          palette = palette)
            ):
            scatter(features[:,i], smoothed_features[:,num_features - 1 - i], color)
            

    
def _plot_scaffold(is_leaf = True, centerline = 0, linecolor = 'lightgrey', linewidth = 1, lineage_name = '',*,
    segment_connection, ax, features, pseudotime, **kwargs):

    (prev_center, prev_maxtime), (curr_center, curr_maxtime) = segment_connection

    ax.vlines(prev_maxtime, ymin = prev_center, ymax = curr_center, color = linecolor, linewidth = linewidth)
    ax.hlines(curr_center, xmin = prev_maxtime, xmax = curr_maxtime, color = linecolor, linewidth = linewidth)

    if is_leaf:
        ax.text(pseudotime.max()*1.02, centerline, lineage_name, fontsize='x-large', ha = 'left')
        
    
def _build_tree(cell_colors = None, size = None, shape = None,*, ax, features, pseudotime, cluster_id, tree_graph, lineage_names, min_pseudotime, plot_fn):

    centerlines = get_dendogram_levels(tree_graph)
    source = get_root_state(tree_graph)
    nx_graph = nx.convert_matrix.from_numpy_array(tree_graph)

    max_times, min_times = {}, {}
    max_times[source] = 0
    min_times[source] = 0
    for i, (start_clus, end_clus) in enumerate(
            [(source, source), *nx.algorithms.traversal.bfs_edges(nx_graph, source)]
        ):

        centerline = centerlines[end_clus]

        segment_mask = cluster_id == lineage_names[end_clus]

        segment_features = features[segment_mask]
        segment_pseudotime = pseudotime[segment_mask]
        segment_cell_colors = None if cell_colors is None else cell_colors[segment_mask]

        if len(segment_features) > 0:
            
            max_times[end_clus] = segment_pseudotime.max()
            min_times[end_clus] = segment_pseudotime.min()
            
            connection = ((centerlines[start_clus], max_times[start_clus]), 
                            (centerlines[end_clus], min_times[end_clus]))

            segment_is_leaf = is_leaf(tree_graph, end_clus)

            '''if segment_is_leaf:
                max_time, min_time = segment_pseudotime.min(), segment_pseudotime.max()
                total_elapsed = max(max_time - min_time, min_pseudotime if is_leaf else 0.)
                segment_pseudotime = minmax_scale(segment_pseudotime) * total_elapsed + min_time'''

            plot_fn(features = segment_features, pseudotime = segment_pseudotime, is_leaf = segment_is_leaf, is_root = i == 1,ax = ax,
                centerline = centerline, lineage_name = lineage_names[end_clus], segment_connection = connection, cell_colors = segment_cell_colors)

    ax.set(ylim = (-0.5, max(centerlines.values()) + 0.75))
    ax.axis('off')
    


def plot_stream(color = 'black', log_pseudotime = True, figsize = (20,10), hue_order = None,
    scale_features = False, center_baseline = True, window_size = 51, palette = 'Set3', ax=None, title = None, show_legend = True,
    max_bar_height = 0.8, hide_feature_threshold = 0.03, linecolor = 'grey', linewidth = 0.2, clip = 10,
    min_pseudotime = 0.05, split = False, plots_per_row = 4, height = 3, aspect = 1.5, style = 'stream', size = 2,
    feature_labels = None, tree_structure = True, scaffold_linecolor = 'lightgrey', scaffold_linewidth = 2,
    group_names = None, *, features, pseudotime, group, tree_graph):

    assert(isinstance(max_bar_height, float) and max_bar_height > 0 and max_bar_height < 1)
    assert(isinstance(features, np.ndarray))
    
    if len(features.shape) == 1:
        features = features[:,np.newaxis]
    assert(len(features.shape) == 2)
    
    num_features = features.shape[-1]
    #assert(np.issubdtype(features.dtype, np.number))
    
    ## NORMALIZE FEATURES ACROSS PLOTS AND SEGMENTS ##
    if not clip is None:
        means, stds = features.mean(0, keepdims = True), features.std(0, keepdims = True)
        clip_min, clip_max = means - clip*stds, means + clip*stds
        features = np.clip(features, clip_min, clip_max)

    features_min, features_max = features.min(0, keepdims = True), features.max(0, keepdims = True)
    
    if scale_features:
        features = (features - features_min)/(features_max - features_min) #scale relative heights of features
    else:
        features = features-features_min 

    features = np.maximum(features, 0) #just make sure no vals are negative

    if style == 'stream':
        features = features/(features.sum(-1).max()) * max_bar_height
    elif style == 'scatter':
        features = features/(features.max(0)) * max_bar_height
    
    ##

    if group_names is None and not tree_graph is None:
        group_names = list(np.arange(tree_graph.shape[-1]).astype(str))

    if feature_labels is None:
        feature_labels = list(np.arange(features.shape[-1]).astype(str))

    if log_pseudotime:
        pseudotime = np.log(pseudotime+ 1)
        min_pseudotime = np.log(min_pseudotime + 1)

    segment_kwargs = dict(
        window_size = window_size, hide_feature_threshold = hide_feature_threshold, center_baseline = center_baseline,
        palette = palette, linecolor = linecolor, linewidth = linewidth, hue_order = hue_order, show_legend = show_legend,
        color = color, size = size, max_bar_height = max_bar_height,
    )
    
    scaffold_kwargs = dict(linecolor = scaffold_linecolor, linewidth = scaffold_linewidth)

    build_tree_kwargs = dict(pseudotime = pseudotime, cluster_id = group, tree_graph = tree_graph,
            lineage_names = group_names, min_pseudotime = min_pseudotime)
    
    if style == 'stream':
        plot_fn = _plot_stream_segment
    elif style == 'scatter':
        plot_fn = _plot_scatter_segment
    elif style == 'swarm':
        plot_fn = _plot_swarm_segment


    def make_plot(ax, features):

        if style == 'swarm':
            build_tree_kwargs['cell_colors'] = _get_swarm_colors(ax = ax, palette = palette, features = features[:,0],
                show_legend = show_legend, hue_order = hue_order)

        scaffold_fn = partial(_plot_scaffold, **scaffold_kwargs)
        segment_fn = partial(plot_fn, feature_labels = feature_labels, **segment_kwargs)

        if tree_structure:
            _build_tree(**build_tree_kwargs, features = features, ax = ax, plot_fn=segment_fn)
            _build_tree(**build_tree_kwargs, features = features, ax = ax, plot_fn= scaffold_fn)

        else:
            segment_fn(features = features, pseudotime = pseudotime, is_leaf = False, ax = ax,
                centerline = 0, lineage_name = '', segment_connection = None)
    

    fig = None
    if num_features == 1 or (not split and style in ['stream','scatter']):
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        make_plot(ax, features)

        if not title is None:
            ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))
        
        return ax
            
    else:
        
        def map_stream(ax, features, label):
            make_plot(ax, features[:, np.newaxis])
            ax.set_title(str(label), fontdict= dict(fontsize = 'x-large'))
            
        fig,ax = map_plot(map_stream, list(zip(features.T, feature_labels)), plots_per_row= plots_per_row, height= height, aspect= aspect)
        if not title is None:
            fig.suptitle(title, fontsize = 16)
        
    plt.tight_layout()

    if not fig is None:
        return fig, ax
    else:
        return ax