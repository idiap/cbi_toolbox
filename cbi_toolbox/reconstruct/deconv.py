from scipy import fft
import numpy as np


def inverse_psf_rfft(psf, shape=None, l=20):
    """
    Computes the real FFT of a regularized inversed 2D PSF (or projected 3D)
    This follows the convention of fft.rfft: only half the spectrum is computed

    Parameters
    ----------
    psf : array [ZXY] or [XY]
        the 2D PSF (if 3D, will be projected on Z axis)
    shape : tuple (int, int), optional
        shape of the full-sized desired PSF
        (if None, will be the same as the PSF), by default None
    l : int, optional
        regularization lambda, by default 20

    Returns
    -------
    array [XY]
        the real FFT of the inverse PSF

    Raises
    ------
    ValueError
        if the PSF has incorrect number of dimensions
    """
    if psf.ndim == 3:
        psf = psf.sum(0)
    elif psf.ndim != 2:
        raise ValueError("Invalid dimensions for PSF: {}".format(psf.ndim))

    if shape is None:
        shape = psf.shape

    psf = fft.ifftshift(psf)
    psf_fft = fft.rfft2(psf, s=shape, overwrite_x=True)

    filt = [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]
    reg = np.abs(fft.rfft2(filt, s=shape)) ** 2

    return psf_fft.conjugate() / (np.abs(psf_fft) ** 2 + l * reg)


def deconvolve_sinogram(sinogram, psf, l=20):
    """
    Deconvolve a sinogram with given PSF

    Parameters
    ----------
    sinogram : array [TPY]
        the blurred sinogram
    psf : array [ZXY] or [XY]
        the PSF used to deconvolve
    l : float, optional
        strength of the regularization

    Returns
    -------
    array [TPY]
        the deconvolved sinogram
    """
    inverse = inverse_psf_rfft(psf, shape=sinogram.shape[1:], l=l)
    s_fft = fft.rfft2(sinogram)

    return fft.irfft2(s_fft * inverse)


if __name__ == '__main__':
    from cbi_toolbox.simu import optics, primitives, imaging
    import cbi_toolbox.splineradon as spl
    import napari

    TEST_SIZE = 64

    s_psf = optics.gaussian_psf(
        numerical_aperture=0.3,
        npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)

    i_psf = inverse_psf_rfft(s_psf)

    psfft = fft.rfft2(s_psf.sum(0))

    sample = primitives.boccia(
        TEST_SIZE, radius=(0.8 * TEST_SIZE) // 2, n_stripes=4)
    s_theta = np.arange(90)

    s_radon = spl.radon(sample, theta=s_theta, pad=False)
    s_fpsopt = imaging.fps_opt(sample, s_psf, theta=s_theta)

    s_deconv = deconvolve_sinogram(s_fpsopt, s_psf)

    with napari.gui_qt():
        viewer = napari.view_image(s_radon)
        viewer.add_image(s_fpsopt)
        viewer.add_image(s_deconv)

        viewer = napari.view_image(fft.fftshift(
            np.abs(i_psf), 0))
        viewer.add_image(fft.irfft2(psfft * i_psf))
