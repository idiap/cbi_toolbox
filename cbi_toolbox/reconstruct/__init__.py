"""
The reconstruct package provides reconstruction algorithms,
as well as preprocessing tools and performance scores.

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


def psnr(ref, target, norm=None, limit=None, in_place=False):
    """
    Computes the Peak Signal-to-Noise Ratio:
    PSNR = 10 log( limit ^ 2 / MSE(ref, target) )

    Parameters
    ----------
    ref : numpy.ndarray
        The ground-truth reference array.
    target : numpy.ndarray
        The reconstructed array.
    norm : str
        Normalize the images before computing snr, default is None.
    limit: float, optional
        The maximum pixel value used for PSNR computation,
        default is None (max(ref)).
    in_place : bool, optional
        Perform normalizations in-place, by default False.

    Returns
    -------
    float
        The PSNR.
    """

    if norm is None:
        pass
    elif norm == "mse":
        target = scale_to_mse(ref, target, in_place)
    else:
        ref = normalize(ref, in_place)
        target = normalize(target, in_place)

    if limit is None:
        limit = ref.max()

    return 10 * np.log10(limit**2 / mse(ref, target))


def mse(ref, target):
    """
    Computes the Mean Squared Error between two arrays

    Parameters
    ----------
    ref : numpy.ndarray
        Reference array.
    target : numpy.ndarray
        Target array.

    Returns
    -------
    float
        The MSE.
    """

    return np.square(np.subtract(ref, target)).mean()


def normalize(image, mode="std", in_place=False):
    """
    Normalize an image according to the given criterion.

    Parameters
    ----------
    image : numpy.ndarray
        Image to normalize, will be modified.
    mode : str, optional
        Type of normalization to use, by default 'std'.
        Allowed: ['std', 'max', 'sum']
    in_place : bool, optional
        Perform computations in-place, by default False.

    Returns
    -------
    array
        The normalized image (same as input).

    Raises
    ------
    ValueError
        For unknown mode.
    """

    if mode == "std":
        f = np.std(image)
    elif mode == "max":
        f = np.max(image)
    elif mode == "sum":
        f = np.sum(image)
    else:
        raise ValueError("Invalid norm: {}".format(mode))

    if not in_place:
        image = image / f
    else:
        image /= f

    return image


def scale_to_mse(ref, target, in_place=False):
    """
    Scale a target array to minimise MSE with reference

    Parameters
    ----------
    ref : numpy.ndarray
        The reference for MSE.
    target : numpy.ndarray
        The array to rescale.
    in_place : bool, optional
        Perform computations in-place, by default False.

    Returns
    -------
    numpy.ndarray
        The rescaled target.
    """

    w = np.sum(ref * target) / np.sum(target**2)

    if not in_place:
        target = target * w
    else:
        target *= w

    return target


def mutual_information(sig_a, sig_b, bins=20):
    """
    Compute the mutual information between two signals.

    Parameters
    ----------
    sig_a : numpy.ndarray
        The first signal
    sig_b : numpy.ndarray
        The second signal
    bins : int, optional
        The number of bins used for probability density estimation, by default 20

    Returns
    -------
    float
        The mutual information
    """

    hist, _, _ = np.histogram2d(sig_a.ravel(), sig_b.ravel(), bins=bins)

    pxy = hist / float(np.sum(hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]

    nonzeros = pxy > 0

    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))
