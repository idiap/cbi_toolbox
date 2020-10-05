from scipy import fft
import numpy as np


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
        what filter to use for regularization, by default 'laplacian'

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

    if mode == 'laplacian':
        filt = [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]
        reg = np.abs(fft.rfft2(filt, s=shape)) ** 2

    else:
        filt1 = [1, -1, 0]
        filt2 = [1, 0, -1]

        reg_x = np.abs(
            fft.fft(filt1, n=shape[0])) ** 2 + np.abs(fft.fft(filt2, n=shape[0])) ** 2

        reg_y = np.abs(
            fft.rfft(filt1, n=shape[1])) ** 2 + np.abs(fft.rfft(filt2, n=shape[1])) ** 2

        reg = reg_x[:, None] + reg_y[None, :]

    return psf_fft.conjugate() / (np.abs(psf_fft) ** 2 + l * reg)


def deconvolve_sinogram(sinogram, psf, **kwargs):
    """
    Deconvolve a sinogram with given PSF

    Parameters
    ----------
    sinogram : array [TPY]
        the blurred sinogram
    psf : array [ZXY] or [XY]
        the PSF used to deconvolve

    Returns
    -------
    array [TPY]
        the deconvolved sinogram
    """
    inverse = inverse_psf_rfft(psf, shape=sinogram.shape[1:], **kwargs)
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
    i_psf_o = inverse_psf_rfft(s_psf, filter='other')

    psfft = fft.rfft2(s_psf.sum(0))

    sample = primitives.boccia(
        TEST_SIZE, radius=(0.8 * TEST_SIZE) // 2, n_stripes=4)
    s_theta = np.arange(90)

    s_radon = spl.radon(sample, theta=s_theta, pad=False)
    s_fpsopt = imaging.fps_opt(sample, s_psf, theta=s_theta)

    s_deconv = deconvolve_sinogram(s_fpsopt, s_psf)
    s_deconv_o = deconvolve_sinogram(s_fpsopt, s_psf, filter='other')

    with napari.gui_qt():
        viewer = napari.view_image(s_radon)
        viewer.add_image(s_fpsopt)
        viewer.add_image(s_deconv)
        viewer.add_image(s_deconv_o)

        viewer = napari.view_image(fft.fftshift(
            np.abs(i_psf), 0), name='Laplacian')
        viewer.add_image(fft.fftshift(np.abs(i_psf_o), 0), name='Other')

        viewer = napari.view_image(fft.irfft2(psfft * i_psf), name='Laplacian')
        viewer.add_image(fft.irfft2(psfft * i_psf_o), name='Other')
