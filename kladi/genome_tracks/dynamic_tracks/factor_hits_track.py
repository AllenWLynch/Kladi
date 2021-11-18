
from kladi.genome_tracks.core import DynamicTrack, TrackController, slugify, fill_resources, fill_default_vizargs
from pygenometracks.tracks import BedTrack
from kladi.genome_tracks.static_tracks import StaticBedTrack
import os
import numpy as np
import pandas as pd
from kladiv2.tools import utils


class MemoryBedTrack(DynamicTrack, BedTrack):

    RULE_NAME = 'bed'

    def __init__(self,*, track_id, regions, **properties):
        
        context = self.get_context()

        self.regions = [context.parse_region(region) for region in regions]

        #print(self.region[:5])

        super().__init__(track_id, 'none', 
            snakemake_properties = dict(),
            visualization_properties = properties
        )

        
    def get_source_name(self):
        return self.get_snakemake_filename('bed','bed')

    def get_target(self):
        return self.get_snakemake_filename('bed','bed')

    def transform_source(self):
        
        with open(self.get_source_name(), 'w') as f:

            for region in sorted(self.regions, key = lambda r : (str(r[0]), int(r[1]))):
                print(*region, sep = '\t', file = f)



class DynamicFactorHits(TrackController):


    @fill_resources('accessibility_adata')
    @fill_default_vizargs(StaticBedTrack)
    def __init__(self,*,track_id, ids, accessibility_adata = None, factor_type = 'motifs', 
        chrom_col = 'chrom', start_col = 'start', end_col = 'end', **properties):

        self.track_id = slugify(track_id)
        self.children = []

        factors = pd.DataFrame(utils.get_factor_meta(
            accessibility_adata, factor_type = factor_type, mask_factors = False)
        ).reset_index().set_index('id')

        assert(isinstance(ids, (list, np.ndarray, str)))

        if isinstance(ids, str):
            ids = np.array([ids])

        indices = factors.loc[ids]['index'].values
        names = factors.loc[ids].name.values

        for idx, _id, name in zip(indices, ids, names):
            
            peak_hits = accessibility_adata.varm[factor_type + '_hits'].T[idx, :].indices

            regions = accessibility_adata.var.iloc[peak_hits][[chrom_col, start_col, end_col]].values.tolist()
                    
            self.children.append(
                MemoryBedTrack(
                    track_id = '{}_{}'.format(str(track_id), str(_id)),
                    regions = [ '{}:{}-{}'.format(*map(str, region)) for region in regions ],
                    title = name,
                    **properties
                )
            )