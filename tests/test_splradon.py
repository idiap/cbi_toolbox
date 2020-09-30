import numpy as np
import unittest

import cbi_toolbox.splineradon as splradon


class TestSplradon(unittest.TestCase):

    def test_dimension(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.default_rng().random(image_dim)
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = splradon.radon(image_3d)
        spline_2d = splradon.radon(image_2d)

        self.assertTrue(np.allclose(
            spline_2d, spline_3d[..., 0], rtol=1e-12, atol=1e-12))

        out_3d = splradon.iradon(image_3d)
        out_2d = splradon.iradon(image_2d)

        np.testing.assert_allclose(
            out_2d, out_3d[..., 0], rtol=1e-12, atol=1e-12)

    def test_contiguous(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.default_rng().random(image_dim)

        image_3d = np.transpose(image_3d, (1, 0, 2))
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = splradon.radon(image_3d)
        spline_2d = splradon.radon(image_2d)

        self.assertTrue(np.allclose(
            spline_2d, spline_3d[..., 0], rtol=1e-9, atol=1e-12))

        out_3d = splradon.iradon(spline_3d)
        out_2d = splradon.iradon(spline_2d)

        np.testing.assert_allclose(
            out_2d, out_3d[..., 0], rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
