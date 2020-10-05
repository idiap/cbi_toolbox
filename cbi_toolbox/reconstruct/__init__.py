"""
The reconstruct module provides reconstruction algorithm,
as well as preprocessing tools and performance scores.
"""

import numpy as np


def psnr(ref, target):
    """
    Computes the Peak Signal-to-Noise Ratio

    Parameters
    ----------
    ref : array
        the ground-truth reference array
    target : array
        the reconstructed array

    Returns
    -------
    float
        the PSNR
    """
    return 10 * np.log10(np.max(target)**2 / mse(ref, target))


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
