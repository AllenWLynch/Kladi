
import matplotlib.pyplot as plt
from kladiv2.plots.base import map_colors, plot_umap
import numpy as np
import kladiv2.core.adata_interface as adi
from sklearn.preprocessing import minmax_scale

def _plot_chromatin_differential(
    ax, 
    expr_pallete = 'Reds', 
    cis_prediction_palette = 'viridis',
    differential_palette = 'coolwarm',
    size = 1.5, *,
    gene_name,
    umap,
    chromatin_differential, 
    expression,
    cis_prediction,
    trans_prediction
):

    plot_umap(umap, chromatin_differential, ax = ax[0], palette = differential_palette,
    size = size, vmin = -3, vmax = 3, title = gene_name + ' Chromatin Differential')

    plot_umap(umap, expression, palette = expr_pallete, ax = ax[1],
        add_legend = True, size = size, title = gene_name + ' Expression',
        edgecolor = 'lightgrey', linewidths = 0.1)

    plot_umap(umap, np.log(cis_prediction), palette = cis_prediction_palette, ax = ax[2],
        size = size, title = gene_name + ' Cis Prediction')

    plot_order = expression.argsort()
    ax[3].scatter(
        trans_prediction[plot_order],
        cis_prediction[plot_order],
        s = 2 * size,
        c = map_colors(
            ax[3], expression[plot_order], palette = expr_pallete, 
            cbar_kwargs = dict(
                location = 'right', pad = 0.1, shrink = 1., aspect = 15, label = 'Expression'
            )
        ),
        edgecolor = 'lightgrey',
        linewidths = 0.15,
    )
    ax[3].set(
        title = gene_name + ' Cis vs Trans Prediction',
        xscale = 'log', yscale = 'log',
        xlabel = 'Trans Prediction',
        ylabel = 'Cis Prediction',
    )
    line_extent = cis_prediction.max()
    ax[3].plot([0,line_extent], [0, line_extent], color = 'grey')
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)

    plt.tight_layout()
    return ax


@adi.wraps_functional(
    adata_extractor = adi.fetch_differential_plot, adata_adder = adi.return_output,
    del_kwargs = ['gene_names','umap','chromatin_differential','expression','cis_prediction', 'trans_prediction']
)
def plot_chromatin_differential(
    expr_pallete = 'Reds', 
    cis_prediction_palette = 'viridis',
    differential_palette = 'coolwarm',
    height = 3,
    aspect = 1.3, 
    size = 1, *,
    gene_names,
    umap,
    chromatin_differential, 
    expression,
    cis_prediction,
    trans_prediction
):

    num_rows = len(gene_names)
    fig, ax = plt.subplots(num_rows, 4, figsize = ( aspect * height * 4, num_rows * height) )

    if num_rows == 1:
        ax = ax[np.newaxis , :]

    for i, data in enumerate(zip(
        gene_names,
        chromatin_differential.T,
        expression.T,
        cis_prediction.T,
        trans_prediction.T,
    )):

        kwargs = dict(zip(
            ['gene_name','chromatin_differential','expression','cis_prediction','trans_prediction'],
            data
        ))

        _plot_chromatin_differential(ax = ax[i,:], umap = umap, expr_pallete = expr_pallete, cis_prediction_palette = cis_prediction_palette,
            size = size, differential_palette = differential_palette, **kwargs)

    return ax