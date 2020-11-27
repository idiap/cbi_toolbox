import numpy as np
import unittest

import cbi_toolbox.splineradon as spl


class TestSplradon(unittest.TestCase):

    def test_dimension(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.default_rng().random(image_dim)
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = spl.radon(image_3d)
        spline_2d = spl.radon(image_2d)

        np.testing.assert_allclose(
            spline_2d, spline_3d[..., 0], rtol=1e-12, atol=1e-12)

        out_3d = spl.iradon(image_3d)
        out_2d = spl.iradon(image_2d)

        np.testing.assert_allclose(
            out_2d, out_3d[..., 0], rtol=1e-12, atol=1e-12)

    def test_contiguous(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.default_rng().random(image_dim)

        image_3d = np.transpose(image_3d, (1, 0, 2))
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = spl.radon(image_3d)
        spline_2d = spl.radon(image_2d)

        np.testing.assert_allclose(
            spline_2d, spline_3d[..., 0], rtol=1e-9, atol=1e-12)

        out_3d = spl.iradon(spline_3d)
        out_2d = spl.iradon(spline_2d)

        np.testing.assert_allclose(
            out_2d, out_3d[..., 0], rtol=1e-9, atol=1e-12)

    def test_padding(self):
        theta = np.arange(10)
        for size in range(5, 25):
            shape = (size, size)
            image = np.random.default_rng().random(shape)

            for circle in (True, False):
                rd = spl.radon(image, theta, circle=circle,
                               b_spline_deg=(0, 0))
                ird = spl.iradon(rd, theta, circle=circle, b_spline_deg=(0, 0))

                np.testing.assert_array_equal(shape, ird.shape)


if __name__ == "__main__":
    unittest.main()
