"""
This module implements the computation of a bspline kernel for FBP interpolation.    
"""

import numpy as np
import scipy


def get_kernel_table(nt, n1, n2, h, s, angles, degree=True):
    """
    Compute the bspline kernel table used for interpolation in FBP.

    Parameters
    ----------
    nt : int
        Number of points in the kernel.
    n1 : int
        Degree of the first spline basis.
    n2 : int
        Degree of the second spline basis.
    h : float
        Pixel sampling step on the first signal.
    s : float
        Pixel sampling step on the second signal.
    angles : array_like
        Angle at which to compute the kernel.
    degree : bool, optional
        Give angles in degrees instead of radians, by default True.

    Returns
    -------
    [type]
        [description]
    """

    pad_fact = 4
    angles = np.atleast_1d(angles)

    if degree:
        angles = np.deg2rad(angles)

    h1 = np.abs(np.sin(angles) * h)
    h2 = np.abs(np.cos(angles) * h)

    a = np.max(h1 * (n1 + 1) / 2 + h2 * (n1 + 1) / 2 + s * (n2 + 1) / 2)

    n2, n3 = n1, n2
    h3 = s

    T = a / (nt - 1)
    dnu = 1 / (T * (pad_fact * nt - 1))
    nu = -1 / (2 * T) + np.arange(pad_fact * nt) * dnu

    trikernel_hat = np.power(np.sinc(np.outer(h1, nu)), (n1 + 1)) * np.power(
        np.sinc(np.outer(h2, nu)), (n2 + 1)) * np.power(np.sinc(np.outer(h3, nu)), (n3 + 1))

    kernel = np.abs(scipy.fft.rfft(trikernel_hat, axis=1, overwrite_x=True))

    table = kernel[:, 0:nt] / (T * nt * pad_fact)

    return table, a