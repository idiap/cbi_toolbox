"""
The files module allows to load and save data files
"""

import json
import os
import numpy as np
import apeer_ometiff_library.io as omeio


def load_ome_tiff(file_path):
    """
    Load an OME tiff file as a numpy array [ZXY]

    Parameters
    ----------
    file_path : string
        the file to load

    Returns
    -------
    array [ZXY]
        the loaded array
    """

    array, xmlstring = omeio.read_ometiff(file_path)

    array = array.squeeze()
    array = np.ascontiguousarray(array.transpose((0, 2, 1)))

    file_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
    metadata_path = '_'.join((file_name, 'metadata.txt'))

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None

    return array, xmlstring, metadata


def save_ome_tiff(file_path, image, xmlstring=None):
    """
    Save numpy array to OME tiff format

    Parameters
    ----------
    file_path : string
        where to save the data
    image : array[ZXY]
        the array to save
    xmlstring : str, optional
        xml metadata, by default None
    """

    image = image.transpose((0, 2, 1))
    image = image[None, :, None, ...]
    omeio.write_ometiff(file_path, image, xmlstring)


if __name__ == '__main__':
    a = np.zeros((10, 20, 30), dtype=float)
    save_ome_tiff('./test', a)

    load, _, _ = load_ome_tiff('./test')
    print(load.shape)
