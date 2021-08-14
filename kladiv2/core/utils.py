
import numpy as np
from scipy.sparse import coo_matrix

def project_sparse_matrix(input_hits, bin_map, num_bins, binarize = False):

    index_converted = input_hits.tocsc()[bin_map[:,0], :].tocoo()

    input_hits = coo_matrix(
        (index_converted.data, (bin_map[index_converted.row, 1], index_converted.col)), 
        shape = (num_bins, input_hits.shape[1]) if not num_bins is None else None 
    ).tocsr()

    if binarize:
        input_hits.data = np.ones_like(input_hits.data)

    return input_hits