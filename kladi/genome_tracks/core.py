from yaml import dump
from pygenometracks.tracks import GenomeTrack
import threading
import os
import unicodedata
import re
import subprocess
import logging
from kladi.core.plot_utils import map_plot
import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.sparse import isspmatrix

class PipelineException(Exception):
    pass

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

def normalize_matrix(m):

    if isspmatrix(m):
        return np.ravel(np.array(m.todense()))
    elif isinstance(m, (list, tuple)):
        return np.array(m)
    else:
        return np.ravel(np.array(m))    
    

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


class GenomeView(Context):

    def __init__(self,*, workdir, regions, titles = None, 
        cores = 1, skip_snakemake = False, **resources):

        self.workdir = workdir
        self.cores = 1
        self.resources = resources
        self.skip_snakemake = skip_snakemake

        if not os.path.isdir(workdir):
            os.mkdir(workdir)

        self.tracks = []

        if isinstance(regions, (list, np.ndarray)):
            self.regions = [self.parse_region(region) for region in regions]
        else:
            self.regions = [self.parse_region(str(region))]

        if isinstance(titles, str):
            self.titles = [titles]
        elif not titles is None:
            assert(isinstance(titles, (list, np.ndarray)))
            assert(len(titles) == len(self.regions))
            self.titles = titles
        else:
            self.titles = ['{}:{}-{}'.format(*region) for region in self.regions]

    def get_resource(self, resource):
        try:
            return self.resources[resource]
        except KeyError:
            raise KeyError('Resource {} not found attached to figure, or passed to track.'.format(resource))

    def recalculate_tracks(self):
        self.recalculate = True

    def get_config_path(self, filename):
        return os.path.join(os.getcwd(), filename)

    @property
    def regions_path(self):
        return os.path.join(self.workdir, 'regions.bed')

    @property
    def snakefile_path(self):
        return os.path.join(
            os.path.dirname(__file__), 'SnakeFile'
        )

    def parse_region(self, region):
        if re.match('chr\d+:\d+-\d+', region):
            chrom, start, end = re.split('[:-]', region)
            return (chrom, start, end)
        else:
            raise NotImplementedError()       


    def __exit__(self, typ, value, traceback):
        super().__exit__(typ, value, traceback)
        self.generate_tracks()

    '''def run_snakemake(self):

        process = subprocess.Popen(['snakemake', '-s', self.snakefile_path, '--cores', str(self.cores),'-q'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while process.stderr.readable():
            line = process.stderr.readline().decode()
            if not line:
                break
            else:
                logging.info(line)

        if not process.poll() == 0:
            raise PipelineException('Error while processing files: ' + process.stderr.read().decode())'''

    @property
    def track_config_path(self):
        return self.get_config_path('tracks_config.ini')


    def generate_tracks(self):
        
        '''
        CHECK IF REGIONS ARE THE SAME!!!!
        if they are, don't mess with track state

        if they aren't, unfreeze all tracks
        '''

        unfreeze = True
        if os.path.isfile(self.regions_path):

            with open(self.regions_path, 'r') as f:
                current_regions = [tuple(x.strip().split('\t')) for x in f.readlines()]

            if current_regions == self.regions:
                logging.info('Regions are the same, will not regenerate tracks.')
                unfreeze = False
        
        if unfreeze:
            self.unfreeze()
            with open(self.regions_path, 'w') as f:
                for region in self.regions:
                    print(*region, sep = '\t', file = f)

        with open(self.get_config_path('config.yaml'), 'w') as f:
            f.write(self.get_snakemake_config())

        for track in self.tracks:
            track.transform_source()

        if not self.skip_snakemake:
            ret = subprocess.run(
                ['snakemake', '-s', self.snakefile_path, '--cores', str(self.cores),'-q'], 
                capture_output=True
            )

            if not ret.returncode == 0:
                raise PipelineException(ret.stderr.decode())

        with open(self.track_config_path, 'w') as f:
            f.write(self.get_track_config())
        logging.info('Track config saved to {}.'.format(self.track_config_path))

    def get_track(self, track_id):

        track_id = slugify(str(track_id))
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        raise ValueError('No track with ID {} exists.'.format(str(track_id)))


    def plot_tracks(self, track_config = None, width = 10, height = None, font_size = None, dpi = 72, 
        track_label_fraction = 0.05, track_label_align = 'left', ext = 'png',
        plots_per_row = None):

        if not os.path.isfile(self.regions_path):
            raise Exception('Must run "generate_tracks" before plotting.')

        if track_config is None:
            track_config = self.track_config_path
        else:
            assert(os.path.isfile(track_config))

        plot_args = [
            'pyGenomeTracks','--tracks', str(track_config), '--BED', self.regions_path,
            '--dpi', str(dpi), '-o', os.path.join(self.workdir, '.' + ext), 
            '--trackLabelFraction', str(track_label_fraction)
        ]
        if not height is None:
            plot_args.extend(['--height', str(height)])
        if not font_size is None:
            plot_args.extend(['--fontSize', str(font_size)])

        ret = subprocess.run(plot_args, capture_output = True)
        if not ret.returncode == 0:
            raise Exception(ret.stderr.decode())

        num_regions = len(self.regions)
        fig, ax = plt.subplots(1, num_regions,
            figsize = (1.5 * width, num_regions * width))
        if num_regions == 1:
            ax = [ax]

        for i, (ax_i, region, title) in enumerate(zip(ax, self.regions, self.titles)):

            img = plt.imread(os.path.join(self.workdir, '_{}-{}-{}.{}'\
                .format(*region, ext)))
            '''pixel_width = img.shape[1]

            if i < (num_regions - 1):
                keepwidth = pixel_width - int(pixel_width * track_label_fraction)
                img = img[:,0:keepwidth,:]'''
            ax_i.imshow(img)
            ax_i.axis('off')
            ax_i.set(title = title)

        plt.tight_layout()

        return fig, ax

    def get_snakemake_config(self):

        config_dict = {'sources' : {}, 'targets' : [],
            'bin' : os.path.join(os.path.dirname(__file__), 'bin')}

        for track in self.tracks:
            if not track.get_target() is None:
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

    def freeze(self):
        for track in self.tracks:
            track.freeze()
        return self

    def unfreeze(self):
        for track in self.tracks:
            track.unfreeze()
        return self

    def __len__(self):
        return self.len(tracks)

def fill_resources(*resources):

    def decorator_fill(func):

        def wrapper(self, *args, **kwargs):
            
            available_resources = self.get_context().resources
            for resource in resources:
                if not resource in kwargs:
                    kwargs[resource] = self.get_context().get_resource(resource)
            
            return func(self, *args, **kwargs)

        return wrapper

    return decorator_fill


class BaseTrack(GenomeTrack):

    RULE_NAME = None

    @classmethod
    def get_properties(cls):
        return cls.DEFAULTS_PROPERTIES

    @classmethod
    def get_context(cls):
        return GenomeView.get_context()

    def __init__(self, track_id, source, 
        visualization_properties = {}, snakemake_properties = {}):

        self.track_id = slugify(track_id)
        self.source = source
        self.source_id = slugify(source)
        self.snakemake_properties = snakemake_properties
        self.recalculate = True

        self.parent = self.get_context()
        self.parent.add_track(self)

        self.set_plot_properties(**visualization_properties)

    def set_plot_properties(self, **properties):

        properties['section_name'] = self.track_id
        properties['file'] = self.get_target()
        properties['file_type'] = self.TRACK_TYPE

        if not 'title' in properties:
            properties['title'] = self.track_id
        if not 'height' in properties:
            properties['height'] = 10

        GenomeTrack.__init__(self, properties)

    def get_track_config(self):
        header = '[' + str(self.track_id) + ']\n'
        properties = '\n'.join([
            '{} = {}'.format(str(prop), str(value))
            for prop, value in self.properties.items() if not value is None
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
        pass

    def freeze(self):
        self.recalculate = False
        return self

    def unfreeze(self):
        self.recalculate = True
        return self

class DynamicTrack(BaseTrack):

    def get_source_name(self):
        pass

    def should_transform(self):
        #restart pipeline if not frozen, or if the source no longer exists
        transforming = self.recalculate or not os.path.isfile(self.get_source_name())
        if transforming:
            logging.info(self.track_id + ': generating new track data...')
        return transforming

    def transform_source(self):
        pass