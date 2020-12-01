"""
The bspline package implements splines and transformations to project signals
onto spline bases.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>,
# Christian Jaques <francois.marelli@idiap.ch>

# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

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
from numpy import zeros, power, multiply
from scipy import signal

from ._change_basis import *
from ._interpolation_conversion import *


def bspline(x, deg):
    """
    B-spline function, faster for degree up to 7.

    Parameters
    ----------
    x : numpy.ndarray
        The points at which the bspline is evaluated.
    deg : int
        The degree of the bspline.

    Returns
    -------
    numpy.ndarray
        The values of the spline for each x.
    """

    if deg == 0:
        return _bspline00(x)
    elif deg == 1:
        return _bspline01(x)
    elif deg == 2:
        return _bspline02(x)
    elif deg == 3:
        return _bspline03(x)
    elif deg == 4:
        return _bspline04(x)
    elif deg == 5:
        return _bspline05(x)
    elif deg == 6:
        return _bspline06(x)
    elif deg == 7:
        return _bspline07(x)
    else:
        return signal.bspline(x, deg)


def _bspline00(x):
    """
    Bspline of degree 0.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    idx = np.abs(x) == 0.5
    y[idx] = 0.5
    idx = np.abs(x) < 0.5
    y[idx] = 1.

    return y


def _bspline01(x):
    """
    Bspline of degree 1.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    idx = np.abs(x) < 1
    y[idx] = 1 - abs(x[idx])

    return y


def _bspline02(x):
    """
    Bspline of degree 2.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    idx = np.abs(x) < 0.5
    y[idx] = 0.75 - power(x[idx], 2)
    idx = (np.abs(x) < 1.5) & (np.abs(x) >= 0.5)
    y[idx] = 0.5 * power((np.abs(x[idx]) - 1.5), 2)

    return y


def _bspline03(x):
    """
    Bspline of degree 3.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    idx = np.abs(x) < 1.
    y[idx] = 0.5 * (power(np.abs(x[idx]), 3) - 2. * power(x[idx], 2)) + 2. / 3.
    idx = (np.abs(x) < 2.) & (np.abs(x) >= 1.)
    y[idx] = -1. / 6. * (power(np.abs(x[idx]) - 2, 3))

    return y


def _bspline04(x):
    """
    Bspline of degree 4.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    a = power(x, 2)
    idx = np.abs(x) < 0.5
    y[idx] = (multiply(a[idx], a[idx] * 0.25 - 5. / 8.) + 115. / 192)
    idx = (np.abs(x) < 1.5) & (np.abs(x) >= 0.5)
    y[idx] = (multiply(np.abs(x[idx]), multiply(np.abs(x[idx]), multiply(np.abs(x[idx]), 5. / 6. - np.abs(
        x[idx]) * 1. / 6.) - 5. / 4.) + 5. / 24.) + 55. / 96.)
    idx = (np.abs(x) < 2.5) & (np.abs(x) >= 1.5)
    a = power(np.abs(x) - 5. / 2., 4)
    y[idx] = (a[idx] * 1. / 24.)

    return y


def _bspline05(x):
    """
    Bspline of degree 5.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    a = power(x, 2)
    idx = np.abs(x) < 1.
    y[idx] = (multiply(a[idx], multiply(a[idx], 1. / 4. -
                                        np.abs(x[idx]) * 1. / 12.) - 1. / 2.) + 11. / 20.0)
    idx = (np.abs(x) < 2.) & (np.abs(x) >= 1.)
    a = np.abs(x)
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], a[
        idx] * 1.0 / 24.0 - 3. / 8.) + 5. / 4.) - 7. / 4.) + 5. / 8.) + 17. / 40.)
    idx = (np.abs(x) < 3.) & (np.abs(x) >= 2.)
    a = 3. - np.abs(x)
    y[idx] = (power(a[idx], 5) * 1. / 120.)

    return y


def _bspline06(x):
    """
    Bspline of degree 6.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    idx = np.abs(x) < 0.5
    a = power(x, 2)
    y[idx] = (multiply(a[idx], multiply(
        a[idx], (7. / 48. - a[idx] * 1. / 36)) - 77. / 192.) + 5887. / 11520.)
    idx = (np.abs(x) < 1.5) & (np.abs(x) >= 0.5)
    a = np.abs(x)
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], a[
        idx] * 1. / 48. - 7. / 48.) + 21. / 64.) - 35. / 288.) - 91. / 256) - 7. / 768.) + 7861. / 15360.)
    idx = (np.abs(x) < 2.5) & (np.abs(x) >= 1.5)
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], 7. / 60. - a[
        idx] * 1. / 120.) - 21. / 32.) + 133. / 72.) - 329. / 128.0) + 1267. / 960.) + 1379. / 7680.)
    idx = (np.abs(x) < 3.5) & (np.abs(x) >= 2.5)
    y[idx] = (power(a[idx] - 7. / 2., 6) * 1. / 720.)

    return y


def _bspline07(x):
    """
    Bspline of degree 7.

    Parameters
    ----------
    x : numpy.ndarray
        The points at whichthe bspline is evaluated.

    Returns
    -------
    numpy.ndarray
        The values of the bspline for each x.
    """

    y = zeros(x.shape)
    a = power(x, 2)
    idx = np.abs(x) < 1.
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], np.abs(
        x[idx]) * 1. / 144. - 1. / 36.) + 1. / 9.) - 1. / 3.) + 151. / 315.)
    idx = (np.abs(x) < 2.) & (np.abs(x) >= 1.)
    a = np.abs(x)
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx],
                                                                                                            1. / 20. -
                                                                                                            a[
                                                                                                                idx] * 1. / 240.) - 7. / 30.) + 1. / 2.) - 7. / 18.) - 1. / 10.) - 7. / 90.) + 103. / 210.)
    idx = (np.abs(x) < 3.) & (np.abs(x) >= 2.)
    y[idx] = (multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], multiply(a[idx], a[
        idx] * 1. / 720. - 1. / 36.) + 7. / 30.) - 19. / 18.) + 49. / 18.) - 23. / 6.) + 217. / 90.) - 139. / 630.)
    idx = (np.abs(x) < 4.) & (np.abs(x) >= 3)
    y[idx] = power(4. - a[idx], 7) * 1. / 5040.

    return y
