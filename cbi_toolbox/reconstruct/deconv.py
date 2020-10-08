import logging
from scipy import fft
import numpy as np
from cbi_toolbox.utils import fft_size


def inverse_psf_rfft(psf, shape=None, l=20, mode='laplacian'):
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
    mode : str, optional
        the regularizer used, by default laplacian

    Returns
    -------
    array [XY]
        the real FFT of the inverse PSF

    Raises
    ------
    ValueError
        if the PSF has incorrect number of dimensions
        if the regularizer is unknown
    """
    if psf.ndim == 3:
        psf = psf.sum(0)
    elif psf.ndim != 2:
        raise ValueError("Invalid dimensions for PSF: {}".format(psf.ndim))

    if shape is None:
        shape = psf.shape

    psf_fft = fft.rfft2(psf, s=shape)

    # We need to shift the PSF so that the center is located at the (0, 0) pixel
    # otherwise deconvolving will shift every pixel
    freq = fft.rfftfreq(shape[1])
    phase_shift = freq * 2 * np.pi * ((psf.shape[1] - 1) // 2)
    psf_fft *= np.exp(1j * phase_shift[None, :])

    freq = fft.fftfreq(shape[0])
    phase_shift = freq * 2 * np.pi * ((psf.shape[0] - 1) // 2)
    psf_fft *= np.exp(1j * phase_shift[:, None])

    if mode == 'laplacian':
        # Laplacian regularization filter to avoid NaN
        filt = [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]
        reg = np.abs(fft.rfft2(filt, s=shape)) ** 2
    elif mode == 'constant':
        reg = 1
    else:
        raise ValueError('Unknown regularizer: {}'.format(mode))

    return psf_fft.conjugate() / (np.abs(psf_fft) ** 2 + l * reg)


def deconvolve_sinogram(sinogram, psf, l=20, mode='laplacian'):
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

    fft_shape = [fft_size(s) for s in sinogram.shape[1:]]

    inverse = inverse_psf_rfft(psf, shape=fft_shape, l=l, mode=mode)

    s_fft = fft.rfft2(sinogram, s=fft_shape)
    i_fft = fft.irfft2(s_fft * inverse, s=fft_shape, overwrite_x=True)

    return i_fft[:, :sinogram.shape[1], :sinogram.shape[2]]


if __name__ == '__main__':
    from cbi_toolbox.simu import optics, primitives, imaging
    import cbi_toolbox.splineradon as spl
    import napari

    TEST_SIZE = 64

    s_psf = optics.gaussian_psf(
        numerical_aperture=0.3,
        npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)

    i_psf = inverse_psf_rfft(s_psf, l=1e-15, mode='constant')
    psfft = fft.rfft2(s_psf.sum(0))
    dirac = fft.irfft2(psfft * i_psf, s=s_psf.shape[1:])

    sample = primitives.boccia(
        TEST_SIZE, radius=(0.8 * TEST_SIZE) // 2, n_stripes=4)
    s_theta = np.arange(90)

    s_radon = spl.radon(sample, theta=s_theta, circle=True)
    s_fpsopt = imaging.fps_opt(sample, s_psf, theta=s_theta)

    s_deconv = deconvolve_sinogram(s_fpsopt, s_psf, l=0)

    print(np.square(s_deconv - s_radon).max()**0.5)

    with napari.gui_qt():
        viewer = napari.view_image(s_radon)
        viewer.add_image(s_fpsopt)
        viewer.add_image(s_deconv)

        viewer = napari.view_image(fft.fftshift(
            np.abs(i_psf), 0), name='inverse PSF FFT')
        viewer.add_image(dirac)
