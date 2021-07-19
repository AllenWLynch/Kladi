from kladi.genome_tracks.core import BaseTrack
from pygenometracks import tracks as pgt_tracks

def _constructor(self,*,track_id, file, **properties):
    BaseTrack.__init__(self, track_id, file, visualization_properties = properties)

static_tracks = {}
for _name, _track in pgt_tracks.__dict__.items():
    try:
        if isinstance(_track, object) and issubclass(_track, pgt_tracks.GenomeTrack):
            static_tracks['Static' + _name] = type(_name, (BaseTrack, _track), {"__init__": _constructor})

    except TypeError:
        pass

locals().update(static_tracks)




