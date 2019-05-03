import cbi_toolbox.csplineradon as cspline
import numpy as np

from cbi_toolbox.bsplines import change_basis
from cbi_toolbox.splineradon import filter_sinogram, spline_kernels


def splradon(image, theta=np.arange(180), angledeg=True, n=None,
             b_spline_deg=(1, 3), sampling_steps=(1, 1),
             center=None, captors_center=None, kernel=None):
    # TODO check what could fail with dimensions
    nx = image.shape[1]
    ny = image.shape[0]
    h = sampling_steps[0]
    s = sampling_steps[1]
    ni = b_spline_deg[0]
    ns = b_spline_deg[1]

    shape = np.array(image.shape)
    nc = 2 * int(np.ceil(np.linalg.norm(shape - np.floor((shape - 1) / 2) - 1))) + 3

    if n is not None:
        s = (nc - 1) / (n - 1)
        nc = n

    if center is None:
        center = int(np.floor((ny - 1) / 2)), int(np.floor((nx - 1) / 2))

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    if kernel is None:
        nt = 200
        kernel = spline_kernels.get_kernel_table(nt, ni, ns, h, s, -theta)

    if angledeg:
        theta = np.deg2rad(theta)

    spline_image = change_basis(image, 'cardinal', 'b-spline', ni, (0, 1),
                                boundary_condition='periodic')

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

    if ns > -1:
        sinogram = change_basis(sinogram, 'dual', 'cardinal', ns, 1, boundary_condition='periodic')

    return sinogram


def spliradon(sinogram, theta=None, angledeg=True, n=None, filter_type='RAM-LAK',
              b_spline_deg=(-1, 1), sampling_steps=(1, 1),
              center=None, captors_center=None, kernel=None):
    nc = sinogram.shape[1]
    na = sinogram.shape[0]

    ni = b_spline_deg[0]
    ns = b_spline_deg[1]
    h = sampling_steps[0]
    s = sampling_steps[1]

    if theta is None:
        theta = np.pi / na
    theta = np.atleast_1d(theta)

    if theta.size == 1 and na > 1:
        theta = np.arange(na) * theta

    nx = 2 * np.floor(sinogram.shape[0] / (2 * np.sqrt(2)))
    ny = nx

    if n is not None:
        s = (nc - 1) / (n - 1)
        nc = n

    if center is None:
        center = int(np.floor((ny - 1) / 2)), int(np.floor((nx - 1) / 2))

    if captors_center is None:
        captors_center = s * (nc - 1) / 2

    if kernel is None:
        nt = 200
        kernel = spline_kernels.get_kernel_table(nt, ni, ns, h, s, -theta)

    sinogram, pre_filter = filter_sinogram.filter_sinogram(sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis(sinogram, 'CARDINAL', 'B-SPLINE', ns, 1,
                                boundary_condition='periodic')

    if angledeg:
        theta = np.deg2rad(theta)

    squeeze = False
    if sinogram.ndim < 3:
        sinogram = sinogram[..., np.newaxis]
        squeeze = True

    image = cspline.iradon(
        sinogram,
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

    if ni > -1:
        image = change_basis(image, 'DUAL', 'CARDINAL', ni, (0, 1), boundary_condition='periodic')

    if theta.size > 1:
        image = image * np.pi / theta.size

    return image
