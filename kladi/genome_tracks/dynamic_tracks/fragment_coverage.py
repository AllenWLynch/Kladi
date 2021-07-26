from kladi.genome_tracks.core import DynamicTrack, slugify, fill_resources
from pygenometracks.tracks import BigWigTrack
from kladi.core.plot_utils import map_colors
import os
import numpy as np
from matplotlib.colors import to_hex
from kladi.genome_tracks.core import normalize_matrix

class FragmentCovTrack(DynamicTrack, BigWigTrack):

    RULE_NAME = 'fragment_coverage'

    @fill_resources('fragment_file','genome_file')
    def __init__(self,*, track_id, fragment_file = None, genome_file = None, 
        barcodes = None, norm_constant = 1e6, **properties):

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

    def get_source_name(self):
        return self.get_snakemake_filename('barcodes','txt')

    def get_target(self):
        return self.get_snakemake_filename('pileup', 'bigwig')

    def transform_source(self):
        if self.should_transform():
            with open(self.get_source_name(), 'w') as f:
                f.write('\n'.join(self.barcodes))
        


class DynamicFragmentCov(DynamicTrack):

    @fill_resources('adata','fragment_file','genome_file')
    def __init__(self,*,track_id, adata = None, fragment_file = None, genome_file = None, groupby = None, palette = 'Set2', 
            color = None, hue_order = None, barcode_col = None, min_cells = 50,
            hue = None, hue_function = np.nanmean, layer = None,
            norm_constant = 1e4, labels = None, overlay_previous = 'no', **properties):
        
        if barcode_col is None:
            barcodes = adata.obs_names.values
        else:
            barcodes = adata.obs[barcode_col].values

        if groupby is None:
            SignalLane(track_id = track_id, fragment_file = fragment_file, barcodes = barcodes, 
                norm_constant = norm_constant, **properties)

        groupby_vals = adata.obs[groupby].values
        groups, counts = np.unique(groupby_vals, return_counts = True)
        groups = sorted(groups)
        num_groups = len(groups)

        if labels is None:
            labels = [str(groupby) + ': ' + str(group) for group in groups]
        else:
            assert(len(labels) == num_groups)

        if hue is None:
            if color is None:
                track_colors = map_colors(None, groups, palette, add_legend = False, hue_order = hue_order)
            else:
                track_colors = [color]*num_groups
        else:
            if hue in adata.var_names:
                if layer is None:
                    hue_scores = adata[:, hue].X
                else:
                    hue_scores = adata[:, hue].layers[layer]
                
                hue_scores = normalize_matrix(hue_scores)
            else:
                try:
                    hue_scores = adata.obs[hue].values
                except KeyError:
                    raise KeyError('{} not in vars or obs.'.format(str(hue)))

            hue_vals = [
                hue_function(hue_scores[groupby_vals == group])
                for group, count in zip(groups, counts)
            ]

            track_colors =  map_colors(None, hue_vals, palette, add_legend = False)
            

        if 'title' in properties:
            properties.pop('title')

        self.children = []
        for i, (group, label, color, group_count) in enumerate(
            zip(groups, labels, track_colors, counts)):
            
            if group_count > min_cells:
                self.children.append(FragmentCovTrack(
                    track_id = str(track_id) + '_' + slugify(str(group)), 
                    fragment_file = fragment_file,
                    genome_file = genome_file,
                    barcodes = barcodes[groupby_vals == group],
                    norm_constant = norm_constant,
                    title = label,
                    color = to_hex(color),
                    overlay_previous = overlay_previous if i > 0 else None,
                    **properties,
                ))

    def freeze(self):
        for child in self.children:
            child.freeze()

        return self
        


        
        

