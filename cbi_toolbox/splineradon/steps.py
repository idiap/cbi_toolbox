import numpy as np

from cbi_toolbox.bsplines import change_basis
import cbi_toolbox.csplineradon as cspline
from cbi_toolbox.splineradon import filter_sinogram
from cbi_toolbox.splineradon import spline_kernels


def splradon_pre(image, b_spline_deg=(1, 3)):
    """
    Pre-processing step for radon transform.

    :param image:
    :param b_spline_deg:
    :return:
    """
    ni = b_spline_deg[0]
    return change_basis(image, 'cardinal', 'b-spline', ni, (0, 1),
                        boundary_condition='periodic')


def splradon_inner(spline_image, theta=np.arange(180), angledeg=True, n=None,
                   b_spline_deg=(1, 3), sampling_steps=(1, 1),
                   center=None, captors_center=None, kernel=None):
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
    :param kernel:
    :return:
    """
    nx = spline_image.shape[1]
    ny = spline_image.shape[0]
    h = sampling_steps[0]
    s = sampling_steps[1]
    ni = b_spline_deg[0]
    ns = b_spline_deg[1]

    shape = np.array(spline_image.shape)
    nc = 2 * int(np.ceil(np.linalg.norm(shape - np.floor((shape - 1) / 2) - 1))) + 3

    if n is not None:
        s = (nc - 1) / (n - 1)
        nc = n

    if center is None:
        center = int(np.floor((ny - 1) / 2)), int(np.floor((nx - 1) / 2))

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    if angledeg:
        theta = np.deg2rad(theta)

    if kernel is None:
        nt = 200
        kernel = spline_kernels.get_kernel_table(nt, ni, ns, h, s, -theta, degree=False)

    squeeze = False
    if spline_image.ndim < 3:
        spline_image = spline_image[..., np.newaxis]
        squeeze = True

    sinogram = cspline.radon(
        spline_image,
        h,
        ni,
        center[1],
        center[0],
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


def splradon_post(sinogram, b_spline_deg=(1, 3)):
    """
    Post-processing for the radon transform.

    :param sinogram:
    :param b_spline_deg:
    :return:
    """
    ns = b_spline_deg[1]

    if ns > -1:
        sinogram = change_basis(sinogram, 'dual', 'cardinal', ns, 1, boundary_condition='periodic')
    return sinogram


def spliradon_pre(sinogram, b_spline_deg=(1, 2), filter_type='RAM-LAK'):
    """
    Pre-processing for the inverse radon transform.

    :param sinogram:
    :param b_spline_deg:
    :param filter_type:
    :return:
    """
    ns = b_spline_deg[1]
    sinogram, pre_filter = filter_sinogram.filter_sinogram(sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis(sinogram, 'CARDINAL', 'B-SPLINE', ns, 1,
                                boundary_condition='periodic')
    return sinogram


def spliradon_inner(sinogram_filtered, theta=None, angledeg=True,
                    b_spline_deg=(1, 2), sampling_steps=(1, 1),
                    center=None, captors_center=None, kernel=None):
    """
    Raw inverse radon transform, requires pre and post-processing. Can be run in parallel by splitting the sinogram and theta.

    :param sinogram_filtered:
    :param theta:
    :param angledeg:
    :param n:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param kernel:
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
        kernel = spline_kernels.get_kernel_table(nt, ni, ns, h, s, -theta, degree=False)

    nx = int(2 * np.floor(nc / (2 * np.sqrt(2))))
    ny = nx

    if center is None:
        center = int(np.floor((ny - 1) / 2)), int(np.floor((nx - 1) / 2))

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    squeeze = False
    if sinogram_filtered.ndim < 3:
        sinogram_filtered = sinogram_filtered[..., np.newaxis]
        squeeze = True

    image = cspline.iradon(
        sinogram_filtered,
        s,
        ns,
        captors_center,
        -theta,
        h,
        ni,
        center[1],
        center[0],
        kernel[0],
        kernel[1],
        nx,
        ny
    )

    if squeeze:
        image = np.squeeze(image)

    return image, theta


def spliradon_post(image, theta, b_spline_deg=(1, 2)):
    """
    Post-processing for the inverse radon transform.

    :param image:
    :param theta:
    :param b_spline_deg:
    :return:
    """

    ni = b_spline_deg[0]
    if ni > -1:
        image = change_basis(image, 'DUAL', 'CARDINAL', ni, (0, 1), boundary_condition='periodic')

    if theta.size > 1:
        image = image * np.pi / theta.size

    return image
