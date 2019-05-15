import numpy as np
import unittest

import cbi_toolbox


class TestSplradon(unittest.TestCase):

    def test_dimension(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.rand(*image_dim)
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = cbi_toolbox.splradon(image_3d)
        spline_2d = cbi_toolbox.splradon(image_2d)

        self.assertTrue(np.allclose(spline_2d, spline_3d[..., 0], rtol=1e-12, atol=1e-12))

        out_3d = cbi_toolbox.spliradon(image_3d)
        out_2d = cbi_toolbox.spliradon(image_2d)

        self.assertTrue(np.allclose(out_2d, out_3d[..., 0], rtol=1e-12, atol=1e-12))

    def test_contiguous(self):
        image_dim = (50, 50, 3)
        image_3d = np.random.rand(*image_dim)

        image_3d = np.transpose(image_3d, (1, 0, 2))
        image_2d = np.copy(image_3d[..., 0])

        spline_3d = cbi_toolbox.splradon(image_3d)
        spline_2d = cbi_toolbox.splradon(image_2d)

        self.assertTrue(np.allclose(spline_2d, spline_3d[..., 0], rtol=1e-9, atol=1e-12))

        out_3d = cbi_toolbox.spliradon(spline_3d)
        out_2d = cbi_toolbox.spliradon(spline_2d)

        self.assertTrue(np.allclose(out_2d, out_3d[..., 0], rtol=1e-9, atol=1e-12))
