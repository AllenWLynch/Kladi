
import warnings
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from math import ceil


def map_plot(func, data, plots_per_row = 3, height =4, aspect = 1.5):

    num_plots = len(data)

    num_rows = ceil(num_plots/plots_per_row)
    plots_per_row = min(plots_per_row, num_plots)

    fig, ax = plt.subplots(num_rows, plots_per_row, figsize = (height*aspect*plots_per_row, height*num_rows))
    if num_plots == 1:
        ax = np.array([[ax]])
    elif num_rows==1:
        ax = ax[np.newaxis, :]

    for ax_i, d in zip(ax.ravel(), data):
        
        func(ax_i, *d)

    for ax_i in ax.ravel()[num_plots:]:
        ax_i.axis('off')

    plt.tight_layout()

    return fig, ax


def map_colors(ax, c, palette, add_legend = True, hue_order = None, 
        legend_kwargs = {}, cbar_kwargs = {}, vmin = None, vmax = None):

    assert(isinstance(c, (np.ndarray, list)))
    
    if isinstance(c, list):
        c = np.array(c)

    if np.issubdtype(c.dtype, np.number):

        colormapper=cm.ScalarMappable(Normalize(
            c.min() if vmin is None else vmin,
            c.max() if vmax is None else vmax), 
            cmap=palette)
        c = colormapper.to_rgba(c)

        if add_legend:
            plt.colorbar(colormapper, ax=ax, **cbar_kwargs)

        return c

    else:
        
        classes = sorted(set(c))

        if isinstance(palette, list):
            num_colors = len(palette)
            palette_obj = lambda i : np.array(palette)[i]
        else:
            palette_obj = cm.get_cmap(palette)
            num_colors = len(palette_obj.colors)

        if num_colors > 24:
            color_scaler = (num_colors-1)/(len(classes)-1)

            color_wheel = palette_obj(
                (color_scaler * np.arange(len(classes))).astype(int) % num_colors
            )
        else:
            color_wheel =palette_obj(np.arange(len(classes)) % num_colors)
        
        if hue_order is None:
            class_colors = dict(zip(classes, color_wheel))
        else:
            assert(len(hue_order) == len(classes))
            class_colors = dict(zip(hue_order, color_wheel))

        c = np.array([class_colors[c_class] for c_class in c])

        if add_legend:
            ax.legend(handles = [
                Patch(color = color, label = str(c_class)) for c_class, color in class_colors.items()
            ], **legend_kwargs)

        return c


def plot_umap(X, hue, palette = 'viridis', projection = '2d', ax = None, figsize= (10,5),
        add_legend = True, hue_order = None, size = 2, title = None, vmin = None, vmax = None,
        **plot_kwargs):
    
    plot_order = hue.argsort()

    if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

    colors = map_colors(ax, hue[plot_order], palette, add_legend=add_legend, hue_order = hue_order, vmin = vmin, vmax = vmax,
            cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5)),
            legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='x-large', fontsize='large',
                        bbox_to_anchor=(1.05, 0.5)))

    ax.scatter(X[plot_order,0], X[plot_order,1], c = colors, s= size, **plot_kwargs)
    ax.axis('off')

    if not title is None:
        ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

    return ax