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


import unittest
import numpy as np
from scipy import signal

from cbi_toolbox import bsplines


def test_convert(signal, degree, tolerance, condition):
    source = signal.copy()
    coeffs = bsplines.convert_to_interpolation_coefficients(signal, degree, tolerance,
                                                            boundary_condition=condition, in_place=True)
    r_signal = bsplines.convert_to_samples(
        coeffs, degree, condition, in_place=True)
    np.testing.assert_allclose(source, r_signal)


def test_basis(data, from_b, to_b, degree, axes, tolerance, condition):
    source = data.copy()
    c_out = bsplines.change_basis(
        data, from_b, to_b, degree, axes, tolerance, condition, in_place=True)
    ar = bsplines.change_basis(
        c_out, to_b, from_b, degree, axes, tolerance, condition, in_place=True)

    np.testing.assert_allclose(source, ar, rtol=tolerance * 10, atol=1e-12)


def test_dims(data, from_b, to_b, degree, axes, tolerance, condition):
    source = data.copy()
    out_1 = bsplines.change_basis(
        data, from_b, to_b, degree, axes, tolerance, condition, in_place=True)
    out_2 = bsplines.change_basis(
        source[..., 0], from_b, to_b, degree, axes, tolerance, condition, in_place=True)

    np.testing.assert_allclose(
        out_1[..., 0], out_2)


def test_dims_conversion(data, degree, axes, tolerance, condition):
    source = data.copy()
    c_1 = bsplines.convert_to_interpolation_coefficients(data, degree, tolerance,
                                                         boundary_condition=condition, in_place=True)
    c_2 = bsplines.convert_to_interpolation_coefficients(source[..., 0], degree, tolerance,
                                                         boundary_condition=condition, in_place=True)

    np.testing.assert_allclose(c_1[..., 0], c_2)


class TestBsplines(unittest.TestCase):
    def test_convert(self):
        tolerance = 1e-15
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size,), (data_size, 2), (data_size, 2, 2)]

        for degree in range(8):
            for size in data_size_list:
                signal = np.random.default_rng().random(size)
                for condition in condition_list:
                    test_convert(signal, degree, tolerance, condition)

    def test_basis(self):
        tolerance = 1e-15
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size, data_size), (data_size, data_size, 2)]
        axes_list = [(0,), (0, 1)]
        bases_list = [('cardinal', 'b-spline'), ('cardinal', 'dual')]

        for degree in range(4):
            for size in data_size_list:
                signal = np.random.default_rng().random(size)
                for condition in condition_list:
                    for axis in axes_list:
                        for base in bases_list:
                            from_b, to_b = base
                            test_basis(signal, from_b, to_b, degree,
                                       axis, tolerance, condition)

    def test_dims(self):
        tolerance = 1e-16
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size, data_size, 2)]
        axes_list = [(0,), (0, 1)]
        bases_list = [('cardinal', 'b-spline'), ('cardinal', 'dual')]

        for degree in range(4):
            for size in data_size_list:
                sig = np.random.default_rng().random(size)
                for condition in condition_list:
                    for axis in axes_list:
                        test_dims_conversion(
                            sig, degree, axis, tolerance, condition)
                        for base in bases_list:
                            from_b, to_b = base
                            test_dims(sig, from_b, to_b, degree,
                                      axis, tolerance, condition)

    def test_splines(self):
        x = np.linspace(-10, 10, 10000)
        for degree in range(9):
            np.testing.assert_allclose(signal.bspline(
                x, degree), bsplines.bspline(x, degree))


if __name__ == '__main__':
    unittest.main()
