"""
The bspline package implements splines and transformations to project signals
onto spline bases based on formulas described in [1].

[1] P. Thévenaz, T. Blu, M. Unser, *"Interpolation Revisited"*, IEEE Transactions
on Medical Imaging, vol. 19, no. 7, pp. 739-758, July 2000.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>,
# Christian Jaques <francois.marelli@idiap.ch>
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
#
# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

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
    y[idx] = 1.0

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
    idx = np.abs(x) < 1.0
    y[idx] = 0.5 * (power(np.abs(x[idx]), 3) - 2.0 * power(x[idx], 2)) + 2.0 / 3.0
    idx = (np.abs(x) < 2.0) & (np.abs(x) >= 1.0)
    y[idx] = -1.0 / 6.0 * (power(np.abs(x[idx]) - 2, 3))

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
    y[idx] = multiply(a[idx], a[idx] * 0.25 - 5.0 / 8.0) + 115.0 / 192
    idx = (np.abs(x) < 1.5) & (np.abs(x) >= 0.5)
    y[idx] = (
        multiply(
            np.abs(x[idx]),
            multiply(
                np.abs(x[idx]),
                multiply(np.abs(x[idx]), 5.0 / 6.0 - np.abs(x[idx]) * 1.0 / 6.0)
                - 5.0 / 4.0,
            )
            + 5.0 / 24.0,
        )
        + 55.0 / 96.0
    )
    idx = (np.abs(x) < 2.5) & (np.abs(x) >= 1.5)
    a = power(np.abs(x) - 5.0 / 2.0, 4)
    y[idx] = a[idx] * 1.0 / 24.0

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
    idx = np.abs(x) < 1.0
    y[idx] = (
        multiply(
            a[idx],
            multiply(a[idx], 1.0 / 4.0 - np.abs(x[idx]) * 1.0 / 12.0) - 1.0 / 2.0,
        )
        + 11.0 / 20.0
    )
    idx = (np.abs(x) < 2.0) & (np.abs(x) >= 1.0)
    a = np.abs(x)
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(
                    a[idx],
                    multiply(a[idx], a[idx] * 1.0 / 24.0 - 3.0 / 8.0) + 5.0 / 4.0,
                )
                - 7.0 / 4.0,
            )
            + 5.0 / 8.0,
        )
        + 17.0 / 40.0
    )
    idx = (np.abs(x) < 3.0) & (np.abs(x) >= 2.0)
    a = 3.0 - np.abs(x)
    y[idx] = power(a[idx], 5) * 1.0 / 120.0

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
    y[idx] = (
        multiply(
            a[idx], multiply(a[idx], (7.0 / 48.0 - a[idx] * 1.0 / 36)) - 77.0 / 192.0
        )
        + 5887.0 / 11520.0
    )
    idx = (np.abs(x) < 1.5) & (np.abs(x) >= 0.5)
    a = np.abs(x)
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(
                    a[idx],
                    multiply(
                        a[idx],
                        multiply(a[idx], a[idx] * 1.0 / 48.0 - 7.0 / 48.0)
                        + 21.0 / 64.0,
                    )
                    - 35.0 / 288.0,
                )
                - 91.0 / 256,
            )
            - 7.0 / 768.0,
        )
        + 7861.0 / 15360.0
    )
    idx = (np.abs(x) < 2.5) & (np.abs(x) >= 1.5)
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(
                    a[idx],
                    multiply(
                        a[idx],
                        multiply(a[idx], 7.0 / 60.0 - a[idx] * 1.0 / 120.0)
                        - 21.0 / 32.0,
                    )
                    + 133.0 / 72.0,
                )
                - 329.0 / 128.0,
            )
            + 1267.0 / 960.0,
        )
        + 1379.0 / 7680.0
    )
    idx = (np.abs(x) < 3.5) & (np.abs(x) >= 2.5)
    y[idx] = power(a[idx] - 7.0 / 2.0, 6) * 1.0 / 720.0

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
    idx = np.abs(x) < 1.0
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(a[idx], np.abs(x[idx]) * 1.0 / 144.0 - 1.0 / 36.0) + 1.0 / 9.0,
            )
            - 1.0 / 3.0,
        )
        + 151.0 / 315.0
    )
    idx = (np.abs(x) < 2.0) & (np.abs(x) >= 1.0)
    a = np.abs(x)
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(
                    a[idx],
                    multiply(
                        a[idx],
                        multiply(
                            a[idx],
                            multiply(a[idx], 1.0 / 20.0 - a[idx] * 1.0 / 240.0)
                            - 7.0 / 30.0,
                        )
                        + 1.0 / 2.0,
                    )
                    - 7.0 / 18.0,
                )
                - 1.0 / 10.0,
            )
            - 7.0 / 90.0,
        )
        + 103.0 / 210.0
    )
    idx = (np.abs(x) < 3.0) & (np.abs(x) >= 2.0)
    y[idx] = (
        multiply(
            a[idx],
            multiply(
                a[idx],
                multiply(
                    a[idx],
                    multiply(
                        a[idx],
                        multiply(
                            a[idx],
                            multiply(a[idx], a[idx] * 1.0 / 720.0 - 1.0 / 36.0)
                            + 7.0 / 30.0,
                        )
                        - 19.0 / 18.0,
                    )
                    + 49.0 / 18.0,
                )
                - 23.0 / 6.0,
            )
            + 217.0 / 90.0,
        )
        - 139.0 / 630.0
    )
    idx = (np.abs(x) < 4.0) & (np.abs(x) >= 3)
    y[idx] = power(4.0 - a[idx], 7) * 1.0 / 5040.0

    return y
