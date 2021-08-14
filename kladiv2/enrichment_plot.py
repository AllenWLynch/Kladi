

import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def compact_string(x, max_wordlen = 4, join_spacer = ' ', sep = ' '):
    return '\n'.join(
        [
            join_spacer.join([x for x in segment if not x == '']) for segment in grouper(x.split(sep), max_wordlen, fillvalue='')
        ]
    )


def enrichment_plot(ax, ontology, results,*, label_genes = [],
    text_color, show_top, barcolor, show_genes, max_genes):

    terms, genes, pvals = [],[],[]
    for result in results[:show_top]:
        
        terms.append(
            compact_string(result['term'])
        )        
        genes.append(' '.join(result['genes'][:max_genes]))
        pvals.append(-np.log10(result['pvalue']))
        
    ax.barh(np.arange(len(terms)), pvals, color=barcolor)
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set(title = ontology, xlabel = '-log10 pvalue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if show_genes:
        for j, p in enumerate(ax.patches):
            _y = p.get_y() + p.get_height() - p.get_height()/3
            ax.text(0.1, _y, compact_string(genes[j], max_wordlen=10, join_spacer = ', '), ha="left", color = text_color)