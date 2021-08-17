
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


def _plot_fill(is_top = False,*, ax, time, fill_top, fill_bottom, color, linecolor, linewidth, alpha = 1.):

    ax.fill_between(time, fill_top, fill_bottom, color = color, alpha = alpha)

    if not linecolor is None:
        ax.plot(time, fill_top, color = linecolor, linewidth = linewidth)
        if is_top:
            ax.plot(time, fill_bottom, color = linecolor, linewidth = linewidth)


def _plot_stream_segment(is_leaf = True, centerline = 0, window_size = 20, hide_feature_threshold = 0.05, center_baseline = True,
        palette = 'Set3', linecolor = 'lightgrey', linewidth = 0.1, feature_labels = None, hue_order = None, show_legend = True,
        color = 'black', min_pseudotime = 0., size=20, lineage_name = '', *, ax, features, pseudotime):

    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_features = features.shape[-1]
    if not feature_labels is None:
        assert(len(feature_labels == num_features))
    else:
        feature_labels = np.arange(num_features).astype(str)

    features = features[np.argsort(pseudotime)] #sort by time
    smoothed_features = savgol_filter(features, min(window_size, len(features)//2), 2, axis = 0) #smooth
    ascending_time = pseudotime[np.argsort(pseudotime)] #sort time

    max_time, min_time = ascending_time[-1], ascending_time[0]
    total_elapsed = max(max_time - min_time, min_pseudotime if is_leaf else 0.)
    ascending_time = minmax_scale(ascending_time) * total_elapsed + min_time

    features = np.where(features < hide_feature_threshold, 0., features)
    
    features = np.cumsum(features, axis=-1)
    
    if center_baseline:
        baseline_adjustment = (features[:,-1]/4)[:, np.newaxis]
    else:
        baseline_adjustment = np.zeros((features.shape[0], 1))

    feature_fill_positions = features - baseline_adjustment + centerline
    feature_bottom = centerline - baseline_adjustment.reshape(-1)

    plot_kwargs = dict(
        ax = ax, time = ascending_time, centerline = centerline, 
        baseline_adjustment = baseline_adjustment, 
        linecolor = linecolor, linewidth = linewidth
    )

    legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5))
    
    if num_features == 1:
        _plot_fill(is_top=True, fill_top= feature_fill_positions[:,0], fill_bottom = feature_bottom, color = color, **plot_kwargs)

    for i, color in enumerate(
            map_colors(ax, feature_labels[::-1], add_legend = show_legend, hue_order = hue_order[::-1], legend_params = legend_params)
        ):
        _plot_fill(is_top= i == 0, fill_top= feature_fill_positions[:, num_features - 1 - i], fill_bottom = feature_bottom, 
            color = color, **plot_kwargs)

    #box in borderlines
    ax.vline(ascending_time[0], ymin = feature_bottom[0, 0], ymax = feature_fill_positions[0, -1], 
        color = linecolor, linewidth = linewidth)

    ax.vline(ascending_time[-1], ymin = feature_bottom[-1, 0], ymax = feature_fill_positions[-1, -1], 
        color = linecolor, linewidth = linewidth)



def _plot_scaffold(is_leaf = True, centerline = 0, linecolor = 'lightgrey', linewidth = 0.1, lineage_name = '',*,
    segment_connection, ax, features, pseudotime):

    (prev_center, prev_maxtime), (curr_center, curr_maxtime) = segment_connection

    ax.vline(prev_maxtime, ymin = prev_center, ymax = curr_center, color = linecolor, linewidth = linewidth)
    ax.hline(curr_center, xmin = prev_maxtime, ymax = curr_maxtime, color = linecolor, linewidth = linewidth)

    if is_leaf:
        ax.text(pseudotime.max()*1.02, centerline, lineage_name, fontsize='x-large', ha = 'left')



def _plot_scatter_segment(is_leaf = True, lineage_name = '', centerline = 0,
        palette = 'Set3', linecolor = 'lightgrey', hue_order = None, line_width = 0.1, feature_labels = None,  show_legend = True,
        color = 'black', size = 20,*, ax, features, pseudotime):

    if (features.shape) == 1:
        features = features[:, np.newaxis]

    num_features = features.shape[-1]
    if not feature_labels is None:
        assert(len(feature_labels == num_features))
    else:
        feature_labels = np.arange(num_features).astype(str)

    legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5))
    
    if num_features == 1:
        ax.scatter(pseudotime, features[:, 0] + centerline, s = size, color = color)

    for i, color in enumerate(
            map_colors(ax, feature_labels[::-1], add_legend = show_legend, hue_order = hue_order[::-1], legend_params = legend_params)
        ):
        ax.scatter(pseudotime, features[:, num_features - 1 - i] + centerline, s = size, color = color)


def _build_tree(*, ax, features, pseudotime, cluster_id, tree_graph, lineage_names, plot_fn):

    centerlines = get_dendogram_levels(tree_graph)
    source = get_root_state(tree_graph)
    nx_graph = nx.convert_matrix.from_numpy_array(tree_graph)

    max_times = {}
    for i, start_clus, end_clus in enumerate(
            [(0, 0), *nx.algorithms.traversal.depth_first_search.dfs_edges(nx_graph, source)]
        ):

        centerline = centerlines[end_clus]

        segment_features = features[cluster_id == lineage_names[end_clus]]
        segment_pseudotime = pseudotime[cluster_id == lineage_names[end_clus]]

        if len(segment_features) > 0:

            max_times[end_clus] = segment_pseudotime.max()
            connection = ((centerlines[start_clus], max_times[start_clus]), 
                            (centerlines[end_clus], max_times[end_clus]))

            plot_fn(features = segment_features, pseudotime = segment_pseudotime, is_leaf = is_leaf(tree_graph), ax = ax,
                centerline = centerline, lineage_name = lineage_names[end_clus], segment_connection = connection)

    ax.axis('off')
    plt.tight_layout()
    ax.set(ylim = (-0.5, max(centerlines.values()) + 0.75))




def _plot_stream(split = False, labels = None, color = 'black', log_pseudotime = True, figsize = (20,10), hue_order = None,
    scale_features = False, center_baseline = True, window_size = 20, palette = 'Set3', ax=None, title = None, show_legend = True,
    max_bar_height = 0.5, hide_feature_threshold = 0.03, linecolor = 'grey', linewidth = 0.1, clip = 10,
    min_pseudotime = 0.05, branch_times = None, lineage_names = None, feature_labels = None, tree_structure = True,
    *, features, pseudotime, cluster_id, tree_graph):

    assert(isinstance(max_bar_height, float) and max_bar_height > 0 and max_bar_height < 1)
    assert(isinstance(features, np.ndarray))
    
    if len(features.shape) == 1:
        features = features[:,np.newaxis]
    assert(len(features.shape) == 2)
    assert(np.issubdtype(features.dtype, np.number))


    means, stds = features.mean(0, keepdims = True), features.std(0, keepdims = True)
    clip_min, clip_max = means - clip*stds, means + clip*stds
    features = np.clip(features, clip_min, clip_max)
    features_min, features_max = features.min(0, keepdims = True), features.max(0, keepdims = True)
    
    if scale_features:
        features = (features - features_min)/(features_max - features_min) #scale relative heights of features
    else:
        features = features-features_min 

    features = np.maximum(features, 0) #just make sure no vals are negative
    features = features/(features.sum(-1).max()) * max_bar_height

    if lineage_names is None and not tree_graph is None:
        lineage_names = list(np.arange(tree_graph.shape[-1]).astype(str))

    if feature_labels is None:
        feature_labels = list(np.arange(features.shape[-1]).astype(str))

    if log_pseudotime:
        pseudotime = np.log(pseudotime + 1)
        min_pseudotime = np.log(min_pseudotime + 1)

    segment_kwargs = dict(
        window_size = window_size, hide_feature_threshold = hide_feature_threshold, log_pseudotime = log_pseudotime, center_baseline = center_baseline,
        palette = palette, linecolor = linecolor, linewidth = linewidth, feature_labels = feature_labels, hue_order = hue_order,
        color = color,  min_pseudotime = min_pseudotime
    )

    scaffold_kwargs = dict(linecolor = linecolor, linewidth = linewidth)

    build_tree_kwargs = dict(features = features, pseudotime = pseudotime, cluster_id = cluster_id, tree_graph = tree_graph,
            lineage_names = lineage_names)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    scaffold_fn = partial(_plot_scaffold, **scaffold_kwargs)
    segment_fn = partial(_plot_stream_segment, **segment_kwargs)

    if tree_structure:
        _build_tree(**build_tree_kwargs, ax = ax, plot_fn= scaffold_fn)
        _build_tree(**build_tree_kwargs, ax = ax, plot_fn= segment_fn)

    else:
        segment_fn(features = features, pseudotime = pseudotime, is_leaf = False, ax = ax,
            centerline = 0, lineage_name = '', segment_connection = None)

    if not title is None:
        ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

    return ax