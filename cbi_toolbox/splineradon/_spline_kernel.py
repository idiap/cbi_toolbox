"""
This module implements the computation of a bspline kernel for FBP interpolation.
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
    (numpy.ndarray, float)
        The kernel lookup table and its maximum element.
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
