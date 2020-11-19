"""
This package implements radon and inverse radon transforms using b-spline interpolation formulas.
"""

import numpy as np

from cbi_toolbox.splineradon import filter_sinogram, spline_kernels
from cbi_toolbox.bsplines import change_basis
import cbi_toolbox.splineradon.cradon as cradon


def is_cuda_available(verbose=False):
    try:
        return cradon.is_cuda_available()
    except RuntimeError as e:
        if verbose:
            print(e)
        return False


def radon(image, theta=np.arange(180), angledeg=True, n=None,
          b_spline_deg=(2, 3), sampling_steps=(1, 1),
          center=None, captors_center=None, circle=False,
          kernel=None, use_cuda=False):
    """
    Perform a radon transform on the image.

    :param image: [z, x, y]
    :param theta:
    :param angledeg:
    :param n:
    :param b_spline_deg: (ni, ns)
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param circle: if the object is contained in the inner circle/cylinder (will produce sinogram with same width as image)
    :param kernel:
    :param use_cuda:
    :return: sinogram [angles, captors, y]
    """

    spline_image = radon_pre(image, b_spline_deg)
    sinogram = radon_inner(spline_image, theta, angledeg, n, b_spline_deg, sampling_steps,
                           center, captors_center, circle, kernel, use_cuda=use_cuda)

    sinogram = radon_post(sinogram, b_spline_deg)

    return sinogram


def iradon(sinogram, theta=None, angledeg=True, filter_type='RAM-LAK',
           b_spline_deg=(2, 3), sampling_steps=(1, 1),
           center=None, captors_center=None, circle=False,
           kernel=None, use_cuda=False):
    """
    Perform an inverse radon transform (backprojection) on the sinogram.

    :param sinogram: [angles, captors, y]
    :param theta:
    :param angledeg:
    :param n:
    :param filter_type:
    :param b_spline_deg: (ni, ns)
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param circle: if radon was computed with circle=True
    :param kernel:
    :param use_cuda:
    :return: image [z, x, y]
    """

    sinogram = iradon_pre(sinogram, b_spline_deg, filter_type, circle)

    image, theta = iradon_inner(sinogram, theta, angledeg, b_spline_deg, sampling_steps,
                                center, captors_center, kernel, use_cuda=use_cuda)
    image = iradon_post(image, theta, b_spline_deg)

    return image


def radon_pre(image, b_spline_deg=(2, 3)):
    """
    Pre-processing step for radon transform.

    :param image:
    :param b_spline_deg:
    :return:
    """
    ni = b_spline_deg[0]
    return change_basis(image, 'cardinal', 'b-spline', ni, (0, 1),
                        boundary_condition='periodic')


def radon_inner(spline_image, theta=np.arange(180), angledeg=True, n=None,
                b_spline_deg=(2, 3), sampling_steps=(1, 1),
                center=None, captors_center=None, circle=False,
                kernel=None, use_cuda=False):
    """
    Raw radon transform, require pre and post-processing. This can be run in parallel by splitting theta.

    :param spline_image:
    :param theta:
    :param angledeg:
    :param n:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param circle:
    :param kernel:
    :param use_cuda:
    :return:
    """
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

    if kernel is None:
        nt = 200
        kernel = spline_kernels.get_kernel_table(
            nt, ni, ns, h, s, -theta, degree=False)

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


def radon_post(sinogram, b_spline_deg=(2, 3)):
    """
    Post-processing for the radon transform.

    :param sinogram:
    :param b_spline_deg:
    :return:
    """
    ns = b_spline_deg[1]

    if ns > -1:
        sinogram = change_basis(sinogram, 'dual', 'cardinal',
                                ns, 1, boundary_condition='periodic', in_place=True)
    return sinogram


def iradon_pre(sinogram, b_spline_deg=(2, 3), filter_type='RAM-LAK', circle=False):
    """
    Pre-processing for the inverse radon transform.

    :param sinogram:
    :param b_spline_deg:
    :param filter_type:
    :param circle:
    :return:
    """
    ns = b_spline_deg[1]

    if circle:
        shape = sinogram.shape[1]
        nc = np.ceil(shape * np.sqrt(2))
        nc += (nc % 2 != shape % 2)
        pad = int((nc - shape) // 2)
        padding = [(0, )] * sinogram.ndim
        padding[1] = (pad, )
        sinogram = np.pad(sinogram, padding)

    sinogram, pre_filter = filter_sinogram.filter_sinogram(
        sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis(sinogram, 'CARDINAL', 'B-SPLINE', ns, 1,
                                boundary_condition='periodic')
    return sinogram


def iradon_inner(sinogram_filtered, theta=None, angledeg=True,
                 b_spline_deg=(2, 3), sampling_steps=(1, 1),
                 center=None, captors_center=None,
                 kernel=None, use_cuda=False):
    """
    Raw inverse radon transform, requires pre and post-processing. 
    Can be run in parallel by splitting the sinogram and theta.

    :param sinogram_filtered:
    :param theta:
    :param angledeg:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param kernel:
    :param use_cuda:
    :return:
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

    if kernel is None:
        nt = 200
        kernel = spline_kernels.get_kernel_table(
            nt, ni, ns, h, s, -theta, degree=False)

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


def iradon_post(image, theta, b_spline_deg=(2, 3)):
    """
    Post-processing for the inverse radon transform.

    :param image:
    :param theta:
    :param b_spline_deg:
    :return:
    """

    ni = b_spline_deg[0]
    if ni > -1:
        image = change_basis(image, 'DUAL', 'CARDINAL', ni,
                             (0, 1), boundary_condition='periodic', in_place=True)

    if theta.size > 1:
        image = image * np.pi / (2 * theta.size)

    return image
