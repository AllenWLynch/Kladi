from kladi.genome_tracks.core import DynamicTrack, slugify, fill_resources
from pygenometracks.tracks import BigWigTrack
from kladi.core.plot_utils import map_colors
import os
import numpy as np
from matplotlib.colors import to_hex
from kladi.genome_tracks.core import normalize_matrix


def regions_overlap(region, region2, min_overlap_proportion = 0):

    def _overlap_distance(min1, max1, min2, max2):
        return max(0, min(max1, max2) - max(min1, min2))

    chrom, start, end = region
    chrom2, start2, end2 = region2
    
    start, end, start2, end2 = list(map(int, [start, end, start2, end2]))

    if chrom == chrom2:
        overlap_dist = _overlap_distance(start, end, start2, end2)
        return overlap_dist > 0 and overlap_dist >= (end - start) * min_overlap_proportion
    else:
        return False


class CisModelTrack(DynamicTrack, BigWigTrack):

    RULE_NAME = 'cis_model'

    @fill_resources('genome_file')
    def __init__(self,*,track_id, cis_model, genome_file = None, bin_size = 100,
        extend = 5, **properties):
        
        self.model_params = cis_model.get_normalized_params()
        self.chrom, self.start, self.end, _, self.strand = cis_model.origin
        self.start = int(self.start)
        self.end = int(self.end)
        self.bin_size = bin_size
        self.extend = extend

        super().__init__(track_id, 'none',
            snakemake_properties = dict(
                genome_file = genome_file
            ),
            visualization_properties = properties
        )

    def get_source_name(self):
        return self.get_snakemake_filename('cis_model', 'bed')

    def get_target(self):
        return self.get_snakemake_filename('cis_model', 'bigwig')
    
    @staticmethod
    def get_rp_value(distance, decay):
        
        if distance < 1500:
            return 1
        
        return 0.5**((distance - 1500) / (1e3 * decay))
    
    def write_function_side(self, f, mod, decay):
        last_distance = 0
        for distance in range(1, int(self.extend * 1e3 * decay), self.bin_size):
            interval = self.start + mod * last_distance, self.start + mod * distance
            print(self.chrom, min(interval), max(interval), self.get_rp_value(distance, decay),
                 sep = '\t', file = f)
            last_distance = distance

    def transform_source(self):
        
        mod = -1
        up_decay, down_decay = self.model_params['logdistance']
        if self.strand == '-':
            mod = 1
        
        with open(self.get_source_name(), 'w') as f:
            self.write_function_side(f, mod, up_decay)
            self.write_function_side(f, -1 * mod, down_decay)                   


class DynamicCisModels(DynamicTrack):

    @fill_resources('genome_file')
    def __init__(self, *, track_id, cis_models, genome_file = None, bin_size = 100, extend = 5,
        overlay_previous = 'yes', **properties):

        regions = self.get_context().regions

        self.children = []
        models_added = 0
        for cis_model in cis_models.gene_models:
            if any([
                regions_overlap(region, cis_model.get_bounds(extend)) for region in regions
            ]):
                self.children.append(
                    CisModelTrack(
                        track_id = '{}_{}'.format(track_id, cis_model.gene),
                        title = 'Cis Models' if models_added > 0 else '',
                        cis_model = cis_model,
                        genome_file = genome_file,
                        bin_size = bin_size,
                        extend = extend,
                        overlay_previous = 'yes' if models_added > 0 else 'no',
                        **properties,
                    )
                )
                models_added+=1

    def freeze(self):
        for child in self.children:
            child.freeze()

        return self