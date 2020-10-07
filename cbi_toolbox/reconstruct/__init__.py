"""
The reconstruct module provides reconstruction algorithm,
as well as preprocessing tools and performance scores.
"""

import numpy as np


def psnr(ref, target, norm=None):
    """
    Computes the Peak Signal-to-Noise Ratio

    Parameters
    ----------
    ref : array
        the ground-truth reference array
    target : array
        the reconstructed array
    norm : str
        normalize the images before computing snr, default is None

    Returns
    -------
    float
        the PSNR
    """

    if norm is None:
        pass
    elif norm == 'std':
        target *= ref.std() / target.std()
    elif norm == 'max':
        target *= ref.max() / target.max()
    elif norm == 'sum':
        target *= ref.sum() / target.sum()
    elif norm == 'mse':
        w = np.sum(ref * target) / np.sum(target ** 2)
        target *= w
    else:
        raise ValueError('Unknown normalization: {}'.format(norm))

    return 10 * np.log10(max(target.max(), ref.max())**2 / mse(ref, target))


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
