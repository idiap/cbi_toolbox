"""
The reconstruct package provides reconstruction algorithms,
as well as preprocessing tools and performance scores.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>

# This file is part of CBI Toolbox.

# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def psnr(ref, target, norm=None, limit=None):
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

    Returns
    -------
    float
        The PSNR.
    """

    if norm is None:
        pass
    elif norm == 'mse':
        scale_to_mse(ref, target)
    else:
        normalize(ref)
        normalize(target)

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


def normalize(image, mode='std'):
    """
    Normalize an image according to the given criterion.

    Parameters
    ----------
    image : numpy.ndarray
        Image to normalize, will be modified.
    mode : str, optional
        Type of normalization to use, by default 'std'.
        Allowed: ['std', 'max', 'sum']

    Returns
    -------
    array
        The normalized image (same as input).

    Raises
    ------
    ValueError
        For unknown mode.
    """

    if mode == 'std':
        f = np.std(image)
    elif mode == 'max':
        f = np.max(image)
    elif mode == 'sum':
        f = np.sum(image)
    else:
        raise ValueError('Invalid norm: {}'.format(mode))

    image /= f
    return image


def scale_to_mse(ref, target):
    """
    Scale a target array to minimise MSE with reference

    Parameters
    ----------
    ref : numpy.ndarray
        The reference for MSE.
    target : numpy.ndarray
        The array to rescale (will be done in-place).

    Returns
    -------
    numpy.ndarray
        The rescaled target.
    """
    w = np.sum(ref * target) / np.sum(target ** 2)
    target *= w

    return target
