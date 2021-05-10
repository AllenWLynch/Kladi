
import os
import sys
import configparser
import subprocess
import requests
import zipfile
import io
import tempfile
import pyfaidx
import numpy as np
from scipy import sparse
import logging
from glob import glob

logger = logging.Logger('kladi.motif_scanning')
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read('kladi/motif_scanning/config.ini')

def validate_peaks(peaks):

    assert(isinstance(peaks, (list, np.ndarray)))
    if isinstance(peaks, np.ndarray):
        assert(peaks.dtype in [np.str, 'S'])
        peaks = peaks.tolist()

    for peak in peaks:
        assert(isinstance(peak, (list,tuple)))
        assert(len(peak) == 3)

        try:
            int(peak[1])
            int(peak[2])
            str(peak[0])
        except ValueError:
            raise Exception('Count not coerce peak {} into (<str chrom>, <int start>, <int end>) format'.format(str(peak)))

    return peaks

def get_motif_glob_str():
    return os.path.join(config.get('data','motifs'), '*.{}'.format(config.get('jaspar','pfm_suffix')))

def convert_jaspar_to_moods_pfm(jaspar_file):

    with open(jaspar_file, 'r') as f:
        jaspar_matrix = f.readlines()

    def strip_extras(jaspar_line):
        return ' '.join(jaspar_line.split(' ')[3:-1])
        
    pfm_matrix = '\n'.join([strip_extras(x) for x in jaspar_matrix[1:]])

    with open(jaspar_file, 'w') as f:
        print(pfm_matrix, end = '', file = f)


def download_jaspar_metadata():

    r = requests.get(config.get('jaspar','metadata_url'))
    if r.ok:
        pass
    else:
        raise Exception('Error downloading motifs database from JASPAR')


def download_jaspar_motifs():

    if not os.path.isdir(config.get('data','root')):
        os.mkdir(config.get('data','root'))
    else:
        pass
        #os.remove(config.get('data','dir'))

    r = requests.get(config.get('jaspar','motifs_url'))

    if r.ok:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(config.get('data','motifs'))

        for jaspar_pfm in os.listdir(config.get('data','motifs')):
            convert_jaspar_to_moods_pfm(os.path.join(config.get('data','motifs'), jaspar_pfm))
        
    else:
        raise Exception('Error downloading motifs database from JASPAR')

def get_peak_sequences(peaks, genome, output_file):

    fa = pyfaidx.Fasta(genome)

    with open(output_file, 'w') as f:
        for i, (chrom,start,end) in enumerate(peaks):

            peak_sequence = fa[chrom][int(start) : int(end)].seq

            print('>{idx}\n{sequence}'.format(
                idx = str(i),
                sequence = peak_sequence.upper()
            ), file = f, end=  '\n')

def list_motif_matrices():

    matrix_dir = config.get('data','motifs')

    if not os.path.isdir(matrix_dir):
        return []

    return list(glob(get_motif_glob_str()))

def get_motif_hits(peak_sequences_file, num_peaks, pvalue_threshold = 0.00005):

    logger.info('Scanning peaks for motif hits with p >= {} ...'.format(str(pvalue_threshold)))

    command = ['moods-dna.py', 
        '-m', get_motif_glob_str(), 
        '-s', peak_sequences_file, 
        '-p', str(pvalue_threshold), 
        '--batch']

    logger.info('Building motif background models ...')
    process = subprocess.Popen(' '.join(command), stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)

    motif_matrices = list_motif_matrices()
    motif_idx_map = dict(zip(motif_matrices, np.arange(len(motif_matrices))))

    motif_indices, peak_indices, scores = [],[],[]
    i=0
    while process.stdout.readable():
        line = process.stdout.readline()

        if not line:
            break
        else:
            if i == 0:
                logger.info('Starting scan ...')
            i+=1

            peak_num, motif, hit_pos, strand, score, site, snp = line.decode().strip().split(',')

            motif_indices.append(motif_idx_map[motif])
            peak_indices.append(peak_num)
            scores.append(float(score))

            if i%100000 == 0:
                logger.info('Found {} motif hits ...'.format(str(i)))

    if not process.poll() == 0:
        raise Exception('Error while canning for motifs: ' + process.stderr.read().decode())

    logger.info('Formatting hits matrix ...')
    return sparse.coo_matrix((scores, (peak_indices, motif_indices)), 
        shape = (num_peaks, len(motif_matrices))).tocsr()


def purge_motif_matrices():
    for matrix in list_motif_matrices():
        os.remove(matrix)


def get_motif_enrichments(peaks, genome, pvalue_threshold = 0.00005):

    peaks = validate_peaks(peaks)

    if len(list_motif_matrices()) == 0:
        download_jaspar_motifs()

        if len(list_motif_matrices()) == 0:
            raise Exception('Problem downloading motifs from JASPAR!')

    temp_fasta = tempfile.TemporaryFile()

    try:
        get_peak_sequences(peaks, genome, temp_fasta)

        hits_matrix = get_motif_hits(temp_fasta, len(peaks), pvalue_threshold = pvalue_threshold)

        #get metadata too

    finally:
        os.remove(temp_fasta)