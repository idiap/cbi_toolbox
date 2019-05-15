import numpy as np
import unittest

from cbi_toolbox import bsplines


def test_convert(signal, degree, tolerance, condition):
    coeffs = bsplines.convert_to_interpolation_coefficients(signal, degree, tolerance,
                                                            boundary_condition=condition)
    r_signal = bsplines.convert_to_samples(coeffs, degree, condition)
    return np.allclose(signal, r_signal, rtol=tolerance * 10, atol=1e-12)


def test_basis(data, from_b, to_b, degree, axes, tolerance, condition):
    c_out = bsplines.change_basis(data, from_b, to_b, degree, axes, tolerance, condition)
    ar = bsplines.change_basis(c_out, to_b, from_b, degree, axes, tolerance, condition)

    return np.allclose(data, ar, rtol=tolerance * 10, atol=1e-12)


def test_dims(data, from_b, to_b, degree, axes, tolerance, condition):
    out_1 = bsplines.change_basis(data, from_b, to_b, degree, axes, tolerance, condition)
    out_2 = bsplines.change_basis(data[..., 0], from_b, to_b, degree, axes, tolerance, condition)

    return np.allclose(out_1[..., 0], out_2, rtol=tolerance, atol=1e-12)


def test_dims_conversion(data, degree, axes, tolerance, condition):
    c_1 = bsplines.convert_to_interpolation_coefficients(data, degree, tolerance,
                                                         boundary_condition=condition)
    c_2 = bsplines.convert_to_interpolation_coefficients(data[..., 0], degree, tolerance,
                                                         boundary_condition=condition)

    return np.allclose(c_1[..., 0], c_2, rtol=tolerance, atol=1e-12)


class TestBsplines(unittest.TestCase):
    def test_convert(self):
        print('Testing sample/interpolation conversion')
        tolerance = 1e-15
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size,), (data_size, 2), (data_size, 2, 2)]

        for degree in range(8):
            print('\tSpline degree: {}'.format(degree))
            for size in data_size_list:
                signal = np.random.rand(*size)
                print('\t\tData size: {}'.format(signal.shape))
                for condition in condition_list:
                    print('\t\t\tCondition: {}'.format(condition))
                    self.assertTrue(test_convert(signal, degree, tolerance, condition))

    def test_basis(self):
        print('Testing basis change')
        tolerance = 1e-15
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size, data_size), (data_size, data_size, 2)]
        axes_list = [(0,), (0, 1)]
        bases_list = [('cardinal', 'b-spline'), ('cardinal', 'dual')]

        for degree in range(4):
            print('\tSpline degree: {}'.format(degree))
            for size in data_size_list:
                signal = np.random.rand(*size)
                print('\t\tData size: {}'.format(signal.shape))
                for condition in condition_list:
                    print('\t\t\tCondition: {}'.format(condition))
                    for axis in axes_list:
                        print('\t\t\t\tAxis: {}'.format(axis))
                        for base in bases_list:
                            print('\t\t\t\t\tBases: {}'.format(base))
                            from_b, to_b = base
                            self.assertTrue(
                                test_basis(signal, from_b, to_b, degree, axis, tolerance, condition))

    def test_dims(self):
        print('Testing dimension stability')
        tolerance = 1e-16
        data_size = 50
        condition_list = ['mirror', 'periodic']
        data_size_list = [(data_size, data_size, 2)]
        axes_list = [(0,), (0, 1)]
        bases_list = [('cardinal', 'b-spline'), ('cardinal', 'dual')]

        for degree in range(4):
            print('\tSpline degree: {}'.format(degree))
            for size in data_size_list:
                signal = np.random.rand(*size)
                print('\t\tData size: {}'.format(signal.shape))
                for condition in condition_list:
                    print('\t\t\tCondition: {}'.format(condition))
                    for axis in axes_list:
                        print('\t\t\t\tAxis: {}'.format(axis))
                        print('\t\t\t\t\tInterpolation')
                        self.assertTrue(test_dims_conversion(signal, degree, axis, tolerance, condition))
                        for base in bases_list:
                            print('\t\t\t\t\tBases: {}'.format(base))
                            from_b, to_b = base
                            self.assertTrue(
                                test_dims(signal, from_b, to_b, degree, axis, tolerance, condition))


if __name__ == '__main__':
    unittest.main()
