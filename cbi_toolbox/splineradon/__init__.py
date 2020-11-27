"""
The splineradon package implements radon and inverse radon transforms using
b-spline interpolation formulas described in [1] using multithreading and GPU
acceleration.

**Conventions:**

arrays follow the ZXY convention, with

    - Z : depth axis (axial, focus axis)
    - X : horizontal axis (lateral)
    - Y : vertical axis (lateral, rotation axis when relevant)

sinograms follow the TPY convention, with

    - T : angles (theta)
    - P : captor axis
    - Y : rotation axis

[1] *S. Horbelt, M. Liebling, M. Unser, "Discretization of the Radon Transform
and of Its Inverse by Spline Convolutions," IEEE Transactions on Medical Imaging,
vol 21, no 4, pp. 363-376, April 2002.*
"""

import numpy as np

from cbi_toolbox.bsplines import change_basis
import cbi_toolbox.splineradon._cradon as cradon
from ._filter_sinogram import *
from ._spline_kernel import *


def is_cuda_available(verbose=False):
    """
    Check if CUDA can be used for computations.

    Parameters
    ----------
    verbose : bool, optional
        Print verbose errors, by default False.

    Returns
    -------
    bool
        Whether CUDA is available.
    """

    try:
        return cradon.is_cuda_available()
    except RuntimeError as e:
        if verbose:
            print(e)
        return False


def radon(image, theta=None, angledeg=True, n=None,
          b_spline_deg=(2, 3), sampling_steps=(1, 1),
          center=None, captors_center=None, circle=False,
          nt=200, use_cuda=False):
    """
    Perform a radon transform on the image.

    Parameters
    ----------
    image : numpy.ndarray [ZXY] or [ZX]
        The input image.
    theta : array_like, optional
        The sinogram angles, by default None.
        If None, uses numpy.arange(180).
    angledeg : bool, optional
        Give angles in degrees instead of radians, by default True.
    n : int, optional
        The number of captors (overrides sampling_steps[1]), by default None.
    b_spline_deg : tuple, optional
        (ni, ns) the degrees of the image and sinogram bspline bases,
        by default (2, 3).
    sampling_steps : tuple, optional
        Pixel sampling steps on the image and the sinogram, by default (1, 1).
    center : tuple, optional
        (z, x) the center of rotation of the image, by default None (centered).
    captors_center : [type], optional
        The position of the center of rotation in the sinogram, by default None.
    circle : bool, optional
        If the object is contained in the inner circle/cylinder (will produce a
        sinogram with same width as the image), by default False.
    nt : int, optional
        Number of points stored in the spline kernel, by default 200.
    use_cuda : bool, optional
        Use CUDA GPU acceleration, by default False.

    Returns
    -------
    numpy.ndarray [TPY]
        The computed sinogram.
    """

    if theta is None:
        theta = np.arange(180)

    spline_image = radon_pre(image, b_spline_deg[0])
    sinogram = radon_inner(spline_image, theta, angledeg, n, b_spline_deg, sampling_steps,
                           center, captors_center, circle, nt, use_cuda=use_cuda)

    sinogram = radon_post(sinogram, b_spline_deg[1])

    return sinogram


def iradon(sinogram, theta=None, angledeg=True, filter_type='RAM-LAK',
           b_spline_deg=(2, 3), sampling_steps=(1, 1),
           center=None, captors_center=None, circle=False,
           nt=200, use_cuda=False):
    """
    Perform a filtered back projection on the sinogram.

    Parameters
    ----------
    sinogram : numpy.ndarray [TPY]
        The input sinogram.
    theta : array_like, optional
        The sinogram angles, by default None.
        If None, uses numpy.arange(180).
    angledeg : bool, optional
        Give angles in degrees instead of radians, by default True.
    filter_type : str, optional
        The type of filter used for FBP, by default 'RAM-LAK'.
        Can be one of ['None', 'Ram-Lak', 'Shepp-Logan', 'Cosine'].
    b_spline_deg : tuple, optional
        (ni, ns) the degrees of the image and sinogram bspline bases,
        by default (2, 3).
    sampling_steps : tuple, optional
        Pixel sampling steps on the image and the sinogram, by default (1, 1).
    center : tuple, optional
        (z, x) the center of rotation of the image, by default None (centered).
    captors_center : [type], optional
        The position of the center of rotation in the sinogram, by default None.
    circle : bool, optional
        If the object is contained in the inner circle/cylinder (will produce a
        sinogram with same width as the image), by default False.
    nt : int, optional
        Number of points stored in the spline kernel, by default 200.
    use_cuda : bool, optional
        Use CUDA GPU acceleration, by default False.

    Returns
    -------
    numpy.ndarray [ZXY]
        The reconstructed image.
    """

    sinogram = iradon_pre(sinogram, b_spline_deg[1], filter_type, circle)

    image, theta = iradon_inner(sinogram, theta, angledeg, b_spline_deg, sampling_steps,
                                center, captors_center, nt, use_cuda=use_cuda)
    image = iradon_post(image, theta, b_spline_deg[0])

    return image


def radon_pre(image, ni=2):
    """
    Pre-processing step for radon transform.
    Projects the image onto a bspline basis.

    Parameters
    ----------
    image : numpy.ndarray [ZXY] or [ZX]
        The raw input image.
    ni : tuple, optional
        The degree of the image bspline basis, by default 2.

    Returns
    -------
    np.ndarray
        The projected image.
    """

    return change_basis(image, 'cardinal', 'b-spline', ni, (0, 1),
                        boundary_condition='periodic')


def radon_inner(spline_image, theta=None, angledeg=True, n=None,
                b_spline_deg=(2, 3), sampling_steps=(1, 1),
                center=None, captors_center=None, circle=False,
                nt=200, use_cuda=False):
    """
    Raw radon transform, requires pre and post-processing (projections onto
    bspline bases). This can be run in parallel by splitting theta.

    Parameters
    ----------
    spline_image : numpy.ndarray
        The input image in a bspline basis.
    theta : array_like, optional
        The sinogram angles, by default None.
        If None, uses numpy.arange(180).
    angledeg : bool, optional
        Give angles in degrees instead of radians, by default True.
    n : int, optional
        The number of captors (overrides sampling_steps[1]), by default None.
    b_spline_deg : tuple, optional
        (ni, ns) the degrees of the image and sinogram bspline bases,
        by default (2, 3).
    sampling_steps : tuple, optional
        Pixel sampling steps on the image and the sinogram, by default (1, 1).
    center : tuple, optional
        (z, x) the center of rotation of the image, by default None (centered).
    captors_center : [type], optional
        The position of the center of rotation in the sinogram, by default None.
    circle : bool, optional
        If the object is contained in the inner circle/cylinder (will produce a
        sinogram with same width as the image), by default False.
    nt : int, optional
        Number of points stored in the spline kernel, by default 200.
    use_cuda : bool, optional
        Use CUDA GPU acceleration, by default False.

    Returns
    -------
    numpy.ndarray [TPY]
        The computed sinogram in a bspline basis.
    """

    if theta is None:
        theta = np.arange(180)

    nz = spline_image.shape[0]
    nx = spline_image.shape[1]
    h = sampling_steps[0]
    s = sampling_steps[1]
    ni = b_spline_deg[0]
    ns = b_spline_deg[1]

    shape = np.max(spline_image.shape[0:2])
    if not circle:
        nc = int(np.ceil(shape * np.sqrt(2)))
        nc += (nc % 2 != shape % 2)

    else:
        nc = shape

    if n is not None:
        s = (nc - 1) / (n - 1)
        nc = n

    if center is None:
        center = (nz - 1) / 2, (nx - 1) / 2

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    if angledeg:
        theta = np.deg2rad(theta)

    kernel = get_kernel_table(nt, ni, ns, h, s, -theta, degree=False)

    squeeze = False
    if spline_image.ndim < 3:
        spline_image = spline_image[..., np.newaxis]
        squeeze = True

    sinogram = cradon.radon(
        spline_image,
        h,
        ni,
        center[0],
        center[1],
        -theta,
        kernel[0],
        kernel[1],
        nc,
        s,
        ns,
        captors_center,
        use_cuda
    )

    if squeeze:
        sinogram = np.squeeze(sinogram)

    return sinogram


def radon_post(sinogram, ns=3):
    """
    Post-processing for the radon transform.
    Projects the sinogram back from a bspline basis.

    Parameters
    ----------
    sinogram : numpy.ndarray
        The sinogram in bspline basis.
    ns : tuple, optional
        The degree of the sinogram bspline basis, by default 3

    Returns
    -------
    numpy.ndarray
        The sinogram.
    """

    if ns > -1:
        sinogram = change_basis(sinogram, 'dual', 'cardinal',
                                ns, 1, boundary_condition='periodic', in_place=True)
    return sinogram


def iradon_pre(sinogram, ns=3, filter_type='RAM-LAK', circle=False):
    """
    Pre-processing for the inverse radon transform.
    Filters the sinogram and projects it onto a bspline basis.

    Parameters
    ----------
    sinogram : numpy.ndarray
        The raw sinogram.
    ns : tuple, optional
        Degree of the sinogram bspline basis, by default 3.
    filter_type : str, optional
        The type of filter used for FBP, by default 'RAM-LAK'.
        Can be one of ['None', 'Ram-Lak', 'Shepp-Logan', 'Cosine'].
    circle : bool, optional
        If the object is contained in the inner circle/cylinder (will produce a
        sinogram with same width as the image), by default False.

    Returns
    -------
    numpy.ndarray
        The filtered and projected sinogram.
    """

    if circle:
        shape = sinogram.shape[1]
        nc = np.ceil(shape * np.sqrt(2))
        nc += (nc % 2 != shape % 2)
        pad = int((nc - shape) // 2)
        padding = [(0, )] * sinogram.ndim
        padding[1] = (pad, )
        sinogram = np.pad(sinogram, padding)

    sinogram, pre_filter = filter_sinogram(sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis(sinogram, 'CARDINAL', 'B-SPLINE', ns, 1,
                                boundary_condition='periodic')
    return sinogram


def iradon_inner(sinogram_filtered, theta=None, angledeg=True,
                 b_spline_deg=(2, 3), sampling_steps=(1, 1),
                 center=None, captors_center=None,
                 nt=200, use_cuda=False):
    """
    Raw inverse radon transform, requires pre and post-processing for sinogram
    filtering and change to bspline basis.
    Can be run in parallel by splitting the sinogram and theta.

    Parameters
    ----------
    sinogram_filtered : numpy.ndarray [TPY]
        The pre-processed sinogram.
    theta : array_like, optional
        The sinogram angles, by default None.
        If None, uses numpy.arange(180).
    angledeg : bool, optional
        Give angles in degrees instead of radians, by default True.
    b_spline_deg : tuple, optional
        (ni, ns) the degrees of the image and sinogram bspline bases,
        by default (2, 3).
    sampling_steps : tuple, optional
        Pixel sampling steps on the image and the sinogram, by default (1, 1).
    center : tuple, optional
        (z, x) the center of rotation of the image, by default None (centered).
    captors_center : [type], optional
        The position of the center of rotation in the sinogram, by default None.
    nt : int, optional
        Number of points stored in the spline kernel, by default 200.
    use_cuda : bool, optional
        Use CUDA GPU acceleration, by default False.

    Returns
    -------
    numpy.ndarray
        The image in bspline basis.
    """

    nc = sinogram_filtered.shape[1]
    na = sinogram_filtered.shape[0]

    ni = b_spline_deg[0]
    ns = b_spline_deg[1]
    h = sampling_steps[0]
    s = sampling_steps[1]

    if theta is None:
        theta = np.pi / na
        angledeg = False

    theta = np.atleast_1d(theta)

    if theta.size == 1 and na > 1:
        theta = np.arange(na) * theta

    if angledeg:
        theta = np.deg2rad(theta)

    kernel = get_kernel_table(nt, ni, ns, h, s, -theta, degree=False)

    nx = int(np.floor(nc / np.sqrt(2)))
    nx -= (nx % 2 != nc % 2)
    nz = nx

    if center is None:
        center = (nz - 1) / 2, (nx - 1) / 2

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    squeeze = False
    if sinogram_filtered.ndim < 3:
        sinogram_filtered = sinogram_filtered[..., np.newaxis]
        squeeze = True

    image = cradon.iradon(
        sinogram_filtered,
        s,
        ns,
        captors_center,
        -theta,
        kernel[0],
        kernel[1],
        nz,
        nx,
        h,
        ni,
        center[0],
        center[1],
        use_cuda
    )

    if squeeze:
        image = np.squeeze(image)

    return image, theta


def iradon_post(image, theta, ni=2):
    """
    Post-processing for the inverse Radon transform.
    Projects the image back from a bspline basis and normalizes it.

    Parameters
    ----------
    image : numpy.ndarray
        The image in bspline basis.
    theta : array_like
        The projection angles of the sinogram.
    ni : tuple, optional
        The degree of the image bspline basis, by default 2.

    Returns
    -------
    numpy.ndarray
        The reconstructed image.
    """

    if ni > -1:
        image = change_basis(image, 'DUAL', 'CARDINAL', ni,
                             (0, 1), boundary_condition='periodic', in_place=True)

    if theta.size > 1:
        image = image * np.pi / (2 * theta.size)

    return image
