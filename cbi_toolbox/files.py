import apeer_ometiff_library.io as omeio
import json
import numpy as np
import os


def load_ome_tiff(file_path):
    '''
    Load an OME tiff file as a numpy array [Z, X, Y]
    :param file_path:
    :return: the C-contiguous array, the xml metadata, the json metadata
    '''

    array, xmlstring = omeio.read_ometiff(file_path)

    array = array.squeeze()
    array = np.ascontiguousarray(array.transpose((0, 2, 1)))

    file_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
    metadata_path = '_'.join((file_name, 'metadata.txt'))

    with open(metadata_path) as f:
        metadata = json.load(f)

    return array, xmlstring, metadata

def save_ome_tiff(file_path, image, xmlstring=None):
    omeio.write_ometiff(file_path, image, xmlstring)
