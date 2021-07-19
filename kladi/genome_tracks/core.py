from yaml import dump
from pygenometracks.tracks import GenomeTrack
import threading
import os
import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '_', value).strip('-_')

class Context(object):

    contexts = threading.local()

    def __enter__(self):

        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")


class Figure(Context):

    def __init__(self,*, workdir, region, recalculate = True):
        self.workdir = workdir

        self.region = region

        if not os.path.isdir(workdir):
            os.mkdir(workdir)

        self.tracks = []

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)
        print('exit_stuff')

    def get_snakemake_config(self):

        config_dict = {'sources' : {}, 'targets' : []}
        for track in self.tracks:
            config_dict['sources'][track.source_id] = track.source
            config_dict['targets'].append(track.get_target())

            config_dict.update(track.get_snakemake_config())

        return dump(config_dict)

    def get_track_config(self):
        return '\n\n'.join(track.get_track_config()
            for track in self.tracks)

    def get_track_ids(self):
        return [track.track_id for track in self.tracks]
        
    def add_track(self, track):
        assert(not track.track_id in self.get_track_ids()), 'Cannot add tracks with duplicate "track_id" attributes'
        self.tracks.append(track)

    def __len__(self):
        return self.len(tracks)


class BaseTrack(GenomeTrack):

    RULE_NAME = None

    @classmethod
    def get_properties(cls):
        return cls.DEFAULTS_PROPERTIES

    def __init__(self, track_id, source, 
        visualization_properties = {}, snakemake_properties = {}):

        self.track_id = slugify(track_id)
        self.source = source
        self.source_id = slugify(source)
        self.snakemake_properties = snakemake_properties

        self.parent = self.get_context()
        self.parent.add_track(self)

        visualization_properties['section_name'] = track_id
        visualization_properties['file'] = self.get_target()
        visualization_properties['file_type'] = self.TRACK_TYPE

        if not 'title' in visualization_properties:
            visualization_properties['title'] = track_id

        GenomeTrack.__init__(self, visualization_properties)

        
    def get_context(self):
        return Figure.get_context()

    def get_track_config(self):
        header = '[' + str(self.track_id) + ']\n'
        properties = '\n'.join([
            '{} = {}'.format(str(prop), str(value))
            for prop, value in self.properties.items()
        ])

        return header + properties + '\n'

    def get_snakemake_filename(self, suffix, ext):
        return os.path.join(self.parent.workdir, '{track_id}-{rulename}-{source_id}-{suffix}.{ext}'.format(
            track_id = self.track_id,
            rulename = self.RULE_NAME,
            source_id = self.source_id,
            suffix = suffix,
            ext = ext
        ))

    def get_snakemake_config(self):
        return {self.track_id : self.snakemake_properties}

    def get_target(self):
        return self.source

    def transform_source(self):
        raise NotImplementedError()