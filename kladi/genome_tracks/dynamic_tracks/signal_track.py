from kladi.genome_tracks.core import BaseTrack
from pygenometracks.tracks import BigWigTrack
from kladi.core.plot_utils import map_colors
import os
import numpy as np
from matplotlib.colors import to_hex

class SignalLane(BaseTrack, BigWigTrack):

    RULE_NAME = 'fragment_signal'

    def __init__(self,*, track_id, fragment_file, genome_file, barcodes = None, norm_constant = 1e5, **properties):

        assert(os.path.isfile(fragment_file))
        assert(os.path.isfile(genome_file))

        super().__init__(
            track_id, fragment_file, 
            snakemake_properties = dict(
                norm_constant = norm_constant, 
                fragment_file = fragment_file,
                genome_file = genome_file),
            visualization_properties = properties,
        )
        
        if barcodes is None:
            barcodes = ['any']
        else:
            assert(isinstance(barcodes, (list, np.ndarray)))

        self.barcodes = barcodes

    def get_bc_filename(self):
        return self.get_snakemake_filename('barcodes','txt')

    def get_target(self):
        return self.get_snakemake_filename('pileup', 'bigwig')

    def transform_source(self):
        with open(self.get_bc_filename(), 'w') as f:
            f.write('\n'.join(self.barcodes))
        

class SignalTrack(BaseTrack, BigWigTrack):

    def __init__(self,*,track_id, adata, fragment_file, genome_file, groupby = None, palette = 'Set3', color = None, hue_order = None, barcode_col = None, min_cells = 50,
            norm_constant = 1e5, titles = None, **properties):

        if barcode_col is None:
            barcodes = adata.obs_names.values
        else:
            barcodes = adata.obs[barcode_col].values

        if groupby is None:
            SignalLane(track_id = track_id, fragment_file = fragment_file, barcodes = barcodes, 
                norm_constant = norm_constant, **properties)

        groupby_vals = adata.obs[groupby].values
        groups, counts = np.unique(groupby_vals, return_counts = True)
        num_groups = len(groups)

        if titles is None:
            titles = [str(track_id) + ': ' + str(group) for group in groups]
        else:
            assert(len(titles) == num_groups)

        if color is None:
            track_colors = map_colors(None, groups, palette, add_legend = False, hue_order = hue_order)
        else:
            track_colors = [color]*num_groups

        if 'title' in properties:
            properties.pop('title')

        for group, title, color, group_count in zip(groups, titles, track_colors, counts):

            if group_count > min_cells:
                SignalLane(
                    track_id = str(track_id) + '_' + str(group), 
                    fragment_file = fragment_file,
                    genome_file = genome_file,
                    barcodes = barcodes[groupby_vals == group],
                    norm_constant = norm_constant,
                    title = title,
                    color = to_hex(color),
                    **properties,
                )


        


        
        

