"""
The utils module provides various utility functions to work with files and arrays
"""

import json
import os
import numpy as np
import apeer_ometiff_library.io as omeio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cbi_toolbox.utils._arrays import *


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


class AnimImshow:
    def __init__(self, images, interval=100):
        self.interval = interval
        self.images = images
        self.fig = plt.figure()
        self.im = plt.imshow(images[0, ...], animated=True)
        self.ani = animation.FuncAnimation(
            self.fig, self._updatefig, interval=interval, frames=images.shape[0], blit=True)

    def _updatefig(self, anim_index, *args):
        self.im.set_array(self.images[anim_index, ...])
        return self.im,

    def save_to_gif(self, path):
        self.ani.save(path, writer='imagemagick', fps=1000/self.interval)


def fft_size(n):
    """
    Returns the smallest power of 2 above n, but no less than 64
    for efficient FFT computations

    Parameters
    ----------
    n : int
        size of the signal
    """

    return max(64, int(2 ** np.ceil(np.log2(n))))
