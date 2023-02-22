"""
The preprocess module implements functions to preprocess experimental images.
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

import cv2
import numpy as np


def erase_corners(image_array, corner_size=300):
    """
    Fill the corners of the image with the closest pixel value.
    This is useful when a diaphragm is visible in the field of view.

    Parameters
    ----------
    image_array : numpy.ndarray
        The original image.
    corner_size : int, optional
        The size of the corner to fill, by default 300

    Returns
    -------
    numpy.ndarray
        The new image with corners filled.
    """

    if corner_size > 0:
        image_array[..., :corner_size, :corner_size] = image_array[
            ..., corner_size, corner_size, None, None
        ]
        image_array[..., :corner_size, -corner_size:] = image_array[
            ..., corner_size, -corner_size, None, None
        ]
        image_array[..., -corner_size:, -corner_size:] = image_array[
            ..., -corner_size, -corner_size, None, None
        ]
        image_array[..., -corner_size:, :corner_size] = image_array[
            ..., -corner_size, corner_size, None, None
        ]

    return image_array


def transmission_to_absorption(image_array, max_value=4096):
    """
    Convert a transmission image into an absorption one.
    This inverses the black-white contrast.

    Parameters
    ----------
    image_array : numpy.ndarray
        The absorption image.
    max_value : float, optional
        The max value for scaling the original image, by default 4096.

    Returns
    -------
    numpy.ndarray
        The absorption contrast image.
    """

    scaled_array = image_array / max_value

    absorption = np.log(1 / scaled_array)

    return absorption


def remove_background_illumination(
    image_array, threshold=0.5, hole_size=250, margin_size=100, border_axis=-1
):
    """
    Removes the background illumination from images using thresholding and
    morphological filtering.

    Parameters
    ----------
    image_array : numpy.ndarray
        The images to be processed as a 3D array, the first dimension iterates
        over the different images.
    threshold : float, optional
        The relative threshold used to detect background, by default 0.5.
    hole_size : int, optional
        Biggest holes removed by filtering, by default 250.
    margin_size : int, optional
        Margin kept around the useful information, by default 100.
    border_axis : int, optional
        Axis used to detect the outer edge of the image, by default -1.

    Returns
    -------
    numpy.ndarray
        The array of images without background illumination.
    """

    mask_bool = image_array < threshold
    mask_int = mask_bool.astype(np.uint8)

    kernel_open = np.ones((hole_size, hole_size), dtype=np.uint8)
    kernel_dilate = np.ones((margin_size, margin_size), dtype=np.uint8)

    if border_axis < 0:
        contour_border_axis = border_axis
    else:
        contour_border_axis = border_axis - 1

    for plane_idx, plane_mask in enumerate(mask_int):
        find_contours = cv2.findContours(
            plane_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )

        major = cv2.__version__.split(".")[0]
        if major == "2":
            contours = find_contours[1]
        else:
            contours = find_contours[0]

        plane_mask.fill(0)

        for index, contour in enumerate(contours):
            if len(contour) > 500 and (
                contour[..., contour_border_axis].max()
                == image_array.shape[border_axis] - 1
                or (contour[..., contour_border_axis].min() == 0)
            ):
                temp_mask = np.zeros_like(mask_int[plane_idx, ...])
                cv2.drawContours(temp_mask, contours, index, 1, cv2.FILLED)

                temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel_open)
                mask_int[plane_idx, ...] |= temp_mask

        mask_int[plane_idx, ...] = cv2.morphologyEx(
            mask_int[plane_idx, ...], cv2.MORPH_ERODE, kernel_dilate
        )

    mask_bool = mask_int.astype(bool)
    image_array[mask_bool] = 0

    return image_array
