
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
    if num_rows==1:
        ax = ax[np.newaxis, :]

    for ax_i, d in zip(ax.ravel(), data):
        
        func(ax_i, *d)

    for ax_i in ax.ravel()[num_plots:]:
        ax_i.axis('off')

    plt.tight_layout()

    return fig, ax


def map_colors(ax, c, palette, add_legend = True, hue_order = None, legend_kwargs = {}, cbar_kwargs = {}):

    assert(isinstance(c, (np.ndarray, list)))
    
    if isinstance(c, list):
        c = np.array(c)

    if np.issubdtype(c.dtype, np.number):

        colormapper=cm.ScalarMappable(Normalize(c.min(), c.max()), cmap=palette)
        c = colormapper.to_rgba(c)

        if add_legend:
            plt.colorbar(colormapper, ax=ax, **cbar_kwargs)

        return c

    else:
        
        classes = sorted(set(c))
        num_colors = len(cm.get_cmap(palette).colors)

        if num_colors > 24:
            color_scaler = (num_colors-1)/(len(classes)-1)

            color_wheel = cm.get_cmap(palette)(
                (color_scaler * np.arange(len(classes))).astype(int) % num_colors
            )
        else:
            color_wheel = cm.get_cmap(palette)(np.arange(len(classes)) % num_colors)
        
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