from kladi.genome_tracks.core import BaseTrack, fill_default_vizargs
from pygenometracks import tracks as pgt_tracks
import copy

def _constructor(self,*,track_id, file = None, **properties):
    BaseTrack.__init__(self, track_id, file, visualization_properties = properties)

static_tracks = {}
for _name, _track in pgt_tracks.__dict__.items():
    try:
        if isinstance(_track, object) and issubclass(_track, pgt_tracks.GenomeTrack):
            static_tracks['Static' + _name] = type(_name, (BaseTrack, _track), {"__init__": fill_default_vizargs(_track)(_constructor) })

    except TypeError as err:
        pass

locals().update(static_tracks)