"""
The utils package provides various utility functions to work with files and
arrays.
"""

import json
import os
import numpy as np
import apeer_ometiff_library.io as omeio

from cbi_toolbox.utils._arrays import *


def load_ome_tiff(file_path):
    """
    Load an OME tiff file as a numpy array [ZXY].

    Parameters
    ----------
    file_path : string
        The file to load.

    Returns
    -------
    array [ZXY]
        The loaded array.
    """

    array, xmlstring = omeio.read_ometiff(file_path)
    array = array.squeeze()
    array = np.ascontiguousarray(array.transpose((0, 2, 1)))

    return array, xmlstring


def save_ome_tiff(file_path, image, xmlstring=None):
    """
    Save numpy array to OME tiff format.

    Parameters
    ----------
    file_path : string
        Where to save the data.
    image : array[ZXY]
        The array to save.
    xmlstring : str, optional
        Xml metadata, by default None.
    """

    image = image.transpose((0, 2, 1))
    image = image[None, :, None, ...]
    omeio.write_ometiff(file_path, image, xmlstring)


def fft_size(n):
    """
    Returns the smallest power of 2 above n, but no less than 64
    for efficient FFT computations.

    Parameters
    ----------
    n : int
        Size of the signal.
    """

    return max(64, int(2 ** np.ceil(np.log2(n))))
