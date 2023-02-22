"""
This module contains tools to read and write ome-tiff files.
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

import numpy as np
import apeer_ometiff_library.io as omeio


def load_ome_tiff(file_path):
    """
    Load an OME tiff file as a numpy array [ZXY].

    Parameters
    ----------
    file_path : string
        The file to load.

    Returns
    -------
    tuple (array [ZXY], str)
        The loaded array, and its xml metadata.

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
