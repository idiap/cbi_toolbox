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


def widefield(obj, psf, pad=False):
    """
    Simulate the widefield imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged
    psf : array [ZXY]
        the PSF of the imaging system
    pad : bool, optional
        extend the field of view to see all contributions, by default False
        if False, the image will have the same size as the object

    Returns
    -------
    [type]
        [description]
    """

    mode = 'full' if pad else 'same'

    image = sig.fftconvolve(obj, psf, mode=mode)
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
        extend the field of view to see all contributions
        (needed if the object is not contained in the inner cylinder to the array), by default False

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

    if pad:
        mode = 'full'
        sinogram = np.empty(
            (theta.size, obj.shape[0] + psf.shape[1] - 1,
             obj.shape[-1] + psf.shape[-1] - 1))
    else:
        mode = 'same'
        sinogram = np.empty(
            (theta.size, obj.shape[0], obj.shape[-1]))

    splimage = ndimage.spline_filter1d(obj, axis=0)
    splimage = ndimage.spline_filter1d(obj, axis=1, output=splimage)

    crop_size = (psf.shape[0] - obj.shape[0]) // 2
    if crop_size:
        psf = psf[crop_size:-crop_size, ...]

    psf = np.flip(psf, 0)

    for idx, angle in enumerate(theta):
        rotated = ndimage.rotate(
            splimage, angle, prefilter=False, reshape=False)

        rotated = sig.fftconvolve(rotated, psf, axes=(1, 2), mode=mode)

        sinogram[idx, ...] = rotated.sum(0)

    return sinogram


def fps_opt(obj, psf, pad=False, **kwargs):
    """
    Simulate the FPS-OPT (focal plane scanning) imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged
    psf : array [ZXY] or array [XY]
        the PSF of the system, or projected PSF along the Z axis
    pad : bool, optional
        pad the vield of view to see all contributions
        (required if the sample is not contained in the inner cylinder of the object), by default False

    **kwargs : 
        to be passed to the radon call

    Returns
    -------
    array [TPY]
        the imaged sinogram

    Raises
    ------
    ValueError
        if the PSF is not 2D or 3D
    """

    if psf.ndim == 3:
        psf = psf.sum(0, keepdims=True)
    elif psf.ndim == 2:
        psf = psf[None, ...]
    else:
        raise ValueError("Invalid dimensions for PSF: {}".format(psf.ndim))

    sinogram = spl.radon(obj, pad=pad, **kwargs)

    mode = 'full' if pad else 'same'

    image = sig.fftconvolve(sinogram, psf, axes=(1, 2), mode=mode)
    return image


def spim(obj, psf, illu, pad=False):
    """
    Simulate the SPIM imaging of an object

    Parameters
    ----------
    obj : array [ZXY]
        the object to be imaged
    psf : array [ZXY]
        the PSF of the imaging objective
    illu : array [ZXY]
        the illumination function of the SPIM
    pad : bool, optional
        pad the vield of view to see all contributions, by default False
    """

    if psf.shape[0] % 2 != illu.shape[0] % 2:
        raise ValueError('In order to correctly center the illumination on the PSF,'
                         ' please profide a PSF with the same Z axis parity as the illumination Z axis')

    if psf.shape[0] < illu.shape[0]:
        raise ValueError(
            'PSF depth must be bigger than SPIM illumination depth (Z axis) - {} < {}'.format(psf.shape[0], illu.shape[0]))

    if illu.shape[1] != obj.shape[1] or illu.shape[2] != obj.shape[2]:
        raise ValueError(
            'SPIM illumination XY must match object XY dimensions - [{},{}] != [{},{}]'.format(
                illu.shape[1], illu.shape[2], obj.shape[1], obj.shape[2]))

    thickness = illu.shape[0]
    crop = (psf.shape[0] - thickness) // 2
    if crop:
        psf = psf[crop:-crop, ...]
    psf = np.flip(psf, 0)

    if pad:
        image = np.empty(
            (obj.shape[0] + thickness - 1, obj.shape[1] + psf.shape[1] - 1, obj.shape[2] + psf.shape[2] - 1))
        mode = 'full'
        obj = np.pad(obj, ((thickness - 1,), (0,), (0,)))

    else:
        image = np.empty(
            (obj.shape[0], obj.shape[1], obj.shape[2]))
        mode = 'same'
        obj = np.pad(obj, ((thickness // 2,), (0,), (0,)))

    for index in range(image.shape[0]):
        sliced = obj[index:index+thickness, ...] * illu
        sliced = sig.fftconvolve(sliced, psf, axes=(1, 2), mode=mode)
        image[index, ...] = sliced.sum(0)

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
    import time

    TEST_SIZE = 64

    sample = primitives.boccia(TEST_SIZE, 4)
    s_psf = optics.gaussian_psf(
        numerical_aperture=0.3,
        npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)
    opt_psf = optics.gaussian_psf(
        numerical_aperture=0.1, npix_axial=TEST_SIZE, npix_lateral=TEST_SIZE+1)
    spim_illu = optics.openspim_illumination(
        slit_opening=2e-3,
        npix_fov=TEST_SIZE, simu_size=4*TEST_SIZE, rel_thresh=1e-3)
    s_theta = np.arange(90)

    start = time.time()
    s_widefield = widefield(sample, s_psf)
    print('Time for widefield: \t{}'.format(time.time() - start))
    start = time.time()
    noisy = noise(s_widefield)
    print('Time for noise: \t{}'.format(time.time() - start))

    start = time.time()
    s_spim = spim(sample, s_psf, spim_illu)
    print('Time for SPIM: \t{}'.format(time.time() - start))

    start = time.time()
    s_opt = opt(sample, opt_psf, theta=s_theta)
    print('Time for OPT: \t{}'.format(time.time() - start))

    start = time.time()
    s_fpsopt = fps_opt(sample, s_psf, theta=s_theta)
    print('Time for FPS-OPT: \t{}'.format(time.time() - start))

    with napari.gui_qt():
        viewer = napari.view_image(sample)
        viewer.add_image(s_widefield)
        viewer.add_image(noisy)
        viewer.add_image(s_opt)
        viewer.add_image(s_fpsopt)
        viewer.add_image(s_spim)
