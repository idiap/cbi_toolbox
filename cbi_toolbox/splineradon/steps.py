import numpy as np

from cbi_toolbox.bsplines import change_basis
from cbi_toolbox.splineradon import filter_sinogram
from cbi_toolbox.splineradon import spline_kernels

import cbi_toolbox.ompradon as ompradon

try:
    import cbi_toolbox.cudaradon as cudaradon
except ImportError:
    cudaradon = None


def radon_pre(image, b_spline_deg=(1, 3)):
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
                b_spline_deg=(1, 3), sampling_steps=(1, 1),
                center=None, captors_center=None, pad=True,
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
    :param pad:
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

    shape = np.array(spline_image.shape)[0:2].max()
    if pad:
        nc = int(np.ceil(shape * np.sqrt(2)))
    else:
        nc = shape

    if n is not None:
        s = (nc - 1) / (n - 1)
        nc = n

    if center is None:
        center = int(np.floor((nz - 1) / 2)), int(np.floor((nx - 1) / 2))

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

    if not use_cuda:
        sinogram = ompradon.radon(
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
            captors_center
        )
    else:
        sinogram = cudaradon.radon_cuda(
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
            captors_center
        )

    if squeeze:
        sinogram = np.squeeze(sinogram)

    return sinogram


def radon_post(sinogram, b_spline_deg=(1, 3)):
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


def iradon_pre(sinogram, b_spline_deg=(1, 2), filter_type='RAM-LAK'):
    """
    Pre-processing for the inverse radon transform.

    :param sinogram:
    :param b_spline_deg:
    :param filter_type:
    :return:
    """
    ns = b_spline_deg[1]
    sinogram, pre_filter = filter_sinogram.filter_sinogram(
        sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis(sinogram, 'CARDINAL', 'B-SPLINE', ns, 1,
                                boundary_condition='periodic')
    return sinogram


def iradon_inner(sinogram_filtered, theta=None, angledeg=True,
                 b_spline_deg=(1, 2), sampling_steps=(1, 1),
                 center=None, captors_center=None, unpad=True,
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
    :param unpad:
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

    if unpad:
        nx = int(np.floor(nc / np.sqrt(2)))
    else:
        nx = nc
    nz = nx

    if center is None:
        center = int(np.floor((nz - 1) / 2)), int(np.floor((nx - 1) / 2))

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    squeeze = False
    if sinogram_filtered.ndim < 3:
        sinogram_filtered = sinogram_filtered[..., np.newaxis]
        squeeze = True

    if not use_cuda:
        image = ompradon.iradon(
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
            center[1]
        )
    else:
        image = cudaradon.iradon_cuda(
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
            center[1]
        )

    if squeeze:
        image = np.squeeze(image)

    return image, theta


def iradon_post(image, theta, b_spline_deg=(1, 2)):
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
        image = image * np.pi / theta.size

    return image
