"""
The imaging module provides simulations of different imaging systems
and acquisition techniques for microscopy

Conventions:
    arrays follow the ZXY convention, with
        Z : depth axis (axial, focus axis)
        X : horizontal axis (lateral)
        Y : vertical axis (lateral, rotation axis when relevant)

    sinograms follow the TPY convention, with
        T : angles (theta)
        P : captor axis
        Y : rotation axis
"""

import scipy.signal as sig
import numpy as np
import numpy.random as random
from scipy import ndimage
from cbi_toolbox import splineradon as spl


def widefield(obj, psf):
    """
    Simulate the widefield imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged
    psf : array [ZXY]
        the PSF of the imaging system

    Returns
    -------
    [type]
        [description]
    """

    image = sig.fftconvolve(obj, psf)
    image.clip(0, None, out=image)
    return image


def opt(obj, psf, theta=np.arange(180), pad=False):
    """
    Simulate the OPT imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged, ZX must be square
    psf : array [ZXY]
        the PSF of the imaging system, Z dimension must match object ZX
    theta : array, optional
        array of rotation angles (in degrees), by default np.arange(180)
    pad : bool, optional
        pad the array so that the sinogram contains the whole object
        (use if the object is not contained in the inner cylinder to the array), by default False

    Returns
    -------
    array [TPY]
        the imaged sinogram

    Raises
    ------
    ValueError
        if the PSF dimension does not match the object
    """

    if not obj.shape[0] == obj.shape[1]:
        raise NotImplementedError(
            'Please provide a square object in ZX dimensions')

    if psf.shape[0] % 2 != obj.shape[0] % 2:
        raise ValueError('In order to correctly center the PSF,'
                         ' please profide a PSF with the same Z axis parity as the object Z axis')

    full_size = obj.shape[0]

    if pad:
        full_size = int(full_size * np.sqrt(2)) + 1
        pad_size = (full_size - obj.shape[0]) // 2
        full_size = obj.shape[0] + 2 * pad_size

    if psf.shape[0] < full_size:
        raise ValueError('PSF Z size must be greater than padded object Z size: {} >= {}'.format(
            psf.shape[0], full_size))

    if pad and pad_size:
        obj = np.pad(obj, ((pad_size,), (pad_size,), (0,)))

    sinogram = np.empty(
        (theta.size, obj.shape[0] + psf.shape[1] - 1,
         obj.shape[-1] + psf.shape[-1] - 1))

    splimage = ndimage.spline_filter1d(obj, axis=0)
    splimage = ndimage.spline_filter1d(obj, axis=1, output=splimage)

    crop_size = (psf.shape[0] - obj.shape[0]) // 2
    if crop_size:
        psf = psf[crop_size:-crop_size, ...]

    psf = np.flip(psf, 0)

    for idx, angle in enumerate(theta):
        rotated = ndimage.rotate(
            splimage, angle, prefilter=False, reshape=False)

        rotated = sig.fftconvolve(rotated, psf, axes=(1, 2))

        sinogram[idx, ...] = rotated.sum(0)

    return sinogram


def fpsopt(obj, psf, **kwargs):
    """
    Simulate the FPS-OPT (focal plane scanning) imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged
    psf : array [ZXY] or array [XY]
        the PSF of the system, or projected PSF along the Z axis

    Returns
    -------
    array [TPY]
        the imaged sinogram

    Raises
    ------
    ValueError
        if the PSF is not 2D or 3D
    """

    radon = spl.radon(obj, **kwargs)

    if psf.ndim == 3:
        psf = psf.sum(0, keepdims=True)
    elif psf.ndim == 2:
        psf = psf[None, ...]
    else:
        raise ValueError("Invalid dimensions for PSF: {}".format(psf.ndim))

    image = sig.fftconvolve(radon, psf, axes=(1, 2))
    return image


def noise(image, photons=150, background=3, seed=None):
    """
    Simulate the acquisition noise of the camera:
        noise = shot noise

    Parameters
    ----------
    image : array [ZXY]
        the clean image
    photons : int, optional
        the max level of photons per pixel, by default 150
    background : int, optional
        the background level of photons per pixel, by default 3
    seed : int, optional
        the seed for rng, by default None

    Returns
    -------
    array [ZXY]
        the noisy image
    """

    image *= (photons - background) / image.max()
    image += background
    rng = random.default_rng(seed)
    poisson = rng.poisson(image)

    return poisson


if __name__ == "__main__":
    from cbi_toolbox.simu import primitives, optics
    import napari

    TEST_SIZE = 128

    sample = primitives.boccia(TEST_SIZE, 4)
    s_psf = optics.gaussian_psf(
        npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)
    opt_psf = optics.gaussian_psf(
        numerical_aperture=0.1, npix_axial=TEST_SIZE, npix_lateral=TEST_SIZE+1)
    spim_illu = optics.openspim_illumination(
        npix_fov=TEST_SIZE, simu_size=4*TEST_SIZE, rel_thresh=1e-6)

    s_widefield = widefield(sample, s_psf)
    noisy = noise(s_widefield)




    s_theta = np.arange(90)
    s_opt = opt(sample, opt_psf, theta=s_theta)
    s_fpsopt = fpsopt(sample, s_psf, theta=s_theta)

    with napari.gui_qt():
        viewer = napari.view_image(sample)
        viewer.add_image(s_widefield)
        viewer.add_image(noisy)
        viewer.add_image(s_opt)
        viewer.add_image(s_fpsopt)
