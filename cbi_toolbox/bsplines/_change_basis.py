"""
This module implements change of basis for bsplines.
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

from ._interpolation_conversion import convert_to_samples, convert_to_interpolation_coefficients
from cbi_toolbox.utils import transpose_dim_to


def change_basis(in_array, from_basis, to_basis, degree, axes=(0,),
                 tolerance=1e-9, boundary_condition='Mirror',
                 in_place=False):
    """
    Change the basis along the provided axes
    (e.g. to convert a 2D signal, use axes=(0, 1)).

    Parameters
    ----------
    in_array : numpy.ndarray
        The input signal.
    from_basis : str
        Source basis, one of ['cardinal', 'dual', b-spline'] (case insensitive).
    to_basis : str
        Target basis, one of ['cardinal', 'dual', b-spline'] (case insensitive).
    degree : int
        Degree of the basis.
    axes : tuple, optional
        Axes on which to compute the change of basis, by default (0,).
    tolerance : float, optional
        Precision of the computations, by default 1e-9.
    boundary_condition : str, optional
        Boundary conditions, one of ['mirror', 'periodic'], by default 'Mirror'.
        (case insensitive)
    in_place : bool, optional
        Overwrite the source data to save memory space, by default False.

    Returns
    -------
    numpy.ndarray
        The signal in the target base.
    """

    if not in_place:
        in_array = in_array.copy()

    try:
        for ax in axes:
            in_array = change_basis(
                in_array, from_basis, to_basis, degree, ax, tolerance, boundary_condition)
    except TypeError:
        in_array = transpose_dim_to(in_array, axes, 0)
        in_array = change_basis_inner(
            in_array, from_basis, to_basis, degree, tolerance, boundary_condition)
        in_array = transpose_dim_to(in_array, 0, axes)

    return in_array


def change_basis_inner(in_array, from_basis, to_basis, degree, tolerance=1e-9,
                       boundary_condition='Mirror'):
    """
    Inner implementation of the change the basis.
    Applies only on the first axis, and always in-place.

    Parameters
    ----------
    in_array : numpy.ndarray
        The input signal.
    from_basis : str
        Source basis, one of ['cardinal', 'dual', b-spline'] (case insensitive).
    to_basis : str
        Target basis, one of ['cardinal', 'dual', b-spline'] (case insensitive).
    degree : int
        Degree of the basis.
    tolerance : float, optional
        Precision of the computations, by default 1e-9.
    boundary_condition : str, optional
        Boundary conditions, one of ['mirror', 'periodic'], by default 'Mirror'.
        (case insensitive)

    Returns
    -------
    numpy.ndarray
        The signal in the target base.
    """

    from_basis = from_basis.upper()
    to_basis = to_basis.upper()

    if from_basis == 'CARDINAL':
        if to_basis == 'CARDINAL':
            return in_array
        elif to_basis == 'B-SPLINE':
            output = convert_to_interpolation_coefficients(in_array, degree, tolerance,
                                                           boundary_condition=boundary_condition)
        elif to_basis == 'DUAL':
            output = change_basis_inner(in_array, 'cardinal', 'b-spline', degree, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
            output = change_basis_inner(output, 'b-spline', 'dual', degree, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    elif from_basis == 'B-SPLINE':
        if to_basis == 'CARDINAL':
            output = convert_to_samples(in_array, degree,
                                        boundary_condition=boundary_condition)
        elif to_basis == 'B-SPLINE':
            return in_array
        elif to_basis == 'DUAL':
            output = change_basis_inner(in_array, 'b-spline', 'cardinal', 2 * degree + 1, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    elif from_basis == 'DUAL':
        if to_basis == 'CARDINAL':
            output = change_basis_inner(in_array, 'dual', 'b-spline', degree, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
            output = change_basis_inner(output, 'b-spline', 'cardinal', degree, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
        elif to_basis == 'B-SPLINE':
            output = change_basis_inner(in_array, 'cardinal', 'b-spline', 2 * degree + 1, tolerance=tolerance,
                                        boundary_condition=boundary_condition)
        elif to_basis == 'DUAL':
            return in_array
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    else:
        raise ValueError("Illegal from_basis : {0}".format(from_basis))

    return output


if __name__ == "__main__":

    data = np.random.default_rng().random((100, 100))
    toleranc = 1e-12
    degre = 3
    axe = (0, 1)
    condition = 'periodic'

    c_out = change_basis(data, 'cardinal', 'dual', degre, axes=axe, tolerance=toleranc, boundary_condition=condition,
                         in_place=False)
    ar = change_basis(c_out, 'dual', 'cardinal', degre, axes=axe,
                      tolerance=toleranc, boundary_condition=condition)

    print('Relative error is ', np.max(
        np.linalg.norm((data - ar) / np.abs(data))))
    print('are samples back to signal? ', np.allclose(data, ar))
