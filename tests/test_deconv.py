import unittest
import numpy as np
from scipy import fft

from cbi_toolbox.simu import primitives, imaging, optics
from cbi_toolbox.reconstruct import deconv
from cbi_toolbox import splineradon as spl


class TestDeconv(unittest.TestCase):
    def setUp(self):
        psf_sizes = [65, 66]
        self.psfs = []

        for s in psf_sizes:
            self.psfs.append(optics.gaussian_psf(
                numerical_aperture=0.3,
                npix_axial=s, npix_lateral=s))

    def test_dirac(self):
        for psf in self.psfs:
            i_psf = deconv.inverse_psf_rfft(psf, l=0)
            psfft = fft.rfft2(psf.sum(0))
            dirac = fft.irfft2(psfft * i_psf, s=psf.shape[1:])

            ref = np.zeros_like(dirac)
            ref[(psf.shape[0] - 1)//2, (psf.shape[0] - 1)//2] = 1

            np.testing.assert_allclose(dirac, ref, atol=1e-12)

    def test_deconv(self):
        size = 100
        sample = primitives.boccia(
            size, radius=(0.8 * size) // 2, n_stripes=4)
        theta = np.arange(90)

        radon = spl.radon(sample, theta=theta, circle=True)

        for psf in self.psfs:
            fpsopt = imaging.fps_opt(sample, psf, theta=theta)
            decon = deconv.deconvolve_sinogram(fpsopt, psf, l=1e-12)

            np.testing.assert_allclose(decon, radon, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
