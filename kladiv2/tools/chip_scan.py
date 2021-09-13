
from kladiv2.tools.motif_scan import validate_peaks, _parse_motif_name
from lisa.core.utils import indices_list_to_sparse_array
from lisa import FromRegions
import kladiv2.core.adata_interface as adi
from functools import partial
import numpy as np


@adi.wraps_functional(
    adata_extractor = adi.get_peaks, 
    adata_adder = partial(adi.add_factor_hits_data, factor_type = 'chip'),
    del_kwargs = ['peaks'])
def get_ChIP_hits_in_peaks(species = 'mm10', *,peaks):

    peaks = validate_peaks(peaks)

    regions_test = FromRegions(species, peaks)

    chip_hits, sample_ids, metadata = regions_test._load_factor_binding_data()

    bin_map = np.hstack([np.arange(chip_hits.shape[0])[:,np.newaxis], regions_test.region_score_map[:, np.newaxis]])

    new_hits = regions_test.data_interface.project_sparse_matrix(chip_hits, bin_map, num_bins=len(peaks))

    hits_matrix = new_hits.T.tocsr()
    factors = metadata['factor']
    parsed_factor_names = list(map(_parse_motif_name, factors))

    return sample_ids, factors, parsed_factor_names, hits_matrix