"""
The utils package provides various utility functions to work with files and
arrays.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause

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
