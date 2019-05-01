import numpy as np
import unittest

from cbi_toolbox import bsplines


def test_convert(signal, degree, tolerance, condition):
    coeffs = bsplines.convert_to_interpolation_coefficients(signal, degree, tolerance, boundary_condition=condition)
    r_signal = bsplines.convert_to_samples(coeffs, degree, condition)

    return np.allclose(signal, r_signal, rtol=tolerance * 10)


class TestBsplines(unittest.TestCase):
    def test_convert(self):
        tolerance = 1e-12
        degree = 3

        signal = np.random.rand(10, 2, 2)
        self.assertTrue(test_convert(signal, degree, tolerance, 'mirror'))
        self.assertTrue(test_convert(signal, degree, tolerance, 'periodic'))

        signal = np.random.rand(10, 2)
        self.assertTrue(test_convert(signal, degree, tolerance, 'mirror'))
        self.assertTrue(test_convert(signal, degree, tolerance, 'periodic'))

        signal = np.random.rand(10)
        self.assertTrue(test_convert(signal, degree, tolerance, 'mirror'))
        self.assertTrue(test_convert(signal, degree, tolerance, 'periodic'))


if __name__ == '__main__':
    unittest.main()
