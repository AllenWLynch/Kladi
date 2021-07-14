from yaml import dump
from pygenometracks.tracks.GenomeTrack import GenomeTrack
from pygenometracks.tracks import 

class Track(GenomeTrack):

    @classmethod
    def get_default_properties(cls):
        return GenomeTrack.DEFAULTS_PROPERTIES

    def __init__(self, workdir, track_id, file, **properties)

        self.track_id = track_id
        properties['file'] = self.get_target(track_id, file)
        super().__init__(properties)

    def get_target(self, track_id, file):
        return file

    def get_snakemake_config(self):
        pass

    def get_track_config(self):
        header = '[{track_id}]\n'.format(self.track_id)
        prop_str = '\n'.join([
            '{} = {}'.format(str(k), str(v)) for k, v in self.properties.items()
        ])

        return header + prop_str

    def prepare_input(self):
        pass

class FragmentCoverageTrack(BigWigTrack, Track):

    def __init__(self, workdir, track_id, file,*,genome, **properties):
        super().__init__(workdir, track_id, file, **properties)
        self.genome = genome

    def get_target(self, track_id, file):
        


        

