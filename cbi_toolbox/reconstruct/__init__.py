"""
The reconstruct module provides reconstruction algorithm,
as well as preprocessing tools and performance scores.
"""

import numpy as np


def psnr(ref, target, norm=None, limit=None):
    """
    Computes the Peak Signal-to-Noise Ratio
    PSNR = 10 log( limit ^ 2 / MSE(ref, target) )

    Parameters
    ----------
    ref : array
        the ground-truth reference array
    target : array
        the reconstructed array
    norm : str
        normalize the images before computing snr, default is None
    limit: float, optional
        the maximum pixel value used for PSNR computation, default is None (max(ref))

    Returns
    -------
    float
        the PSNR
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
    ref : array
        reference array
    target : array
        target array

    Returns
    -------
    float
        the MSE
    """

    return np.square(np.subtract(ref, target)).mean()


def normalize(image, mode='std'):
    """
    Normalize an image according to the given criterion

    Parameters
    ----------
    image : array
        image to normalize, will be modified
    mode : str, optional
        type of normalization to use, by default 'std'

    Returns
    -------
    array
        the normalized image (same as input)

    Raises
    ------
    ValueError
        for unknown mode
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
    ref : array
        the reference for MSE
    target : array
        the array to rescale (will be done in-place)

    Returns
    -------
    array
        the rescaled target
    """
    w = np.sum(ref * target) / np.sum(target ** 2)
    target *= w

    return target
