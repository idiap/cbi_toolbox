import cbi_toolbox.csplineradon as cspline
import numpy as np

from cbi_toolbox.splineradon import change_basis
from cbi_toolbox.splineradon import change_basis_columns
from cbi_toolbox.splineradon import filter_sinogram


def fft_trikernel(nt, a, n1, n2, n3, h1, h2, h3, pad_fact):
    T = a / (nt - 1)
    # TODO check this with Michael
    dnu = 1 / (T * (pad_fact * nt - 1))
    nu = -1 / (2 * T) + np.arange(pad_fact * nt) * dnu

    trikernel_hat = np.power(np.sinc(np.outer(nu, h1)), (n1 + 1)) * np.power(
        np.sinc(np.outer(nu, h2)), (n2 + 1)) * np.power(np.sinc(np.outer(nu, h3)), (n3 + 1))

    kernel = np.abs(np.fft.fft(trikernel_hat, axis=0))

    return kernel[0:nt, :] / (T * nt * pad_fact)


def get_kernel_table(nt, n1, n2, h, s, angles, degree=True):
    pad_fact = 4
    angles = np.atleast_1d(angles)

    if degree:
        angles = np.deg2rad(angles)

    h1 = np.abs(np.sin(angles) * h)
    h2 = np.abs(np.cos(angles) * h)

    a = np.max(h1 * (n1 + 1) / 2 + h2 * (n1 + 1) / 2 + s * (n2 + 1) / 2)

    table = fft_trikernel(nt, a, n1, n1, n2, h1, h2, s, pad_fact)

    return table, a


def splradon(image, theta=np.arange(180), angledeg=True, n=None,
             b_spline_deg=(1, 3), sampling_steps=(1, 1),
             center=None, captors_center=None, kernel=None):
    nx = image.shape[1]
    ny = image.shape[0]
    h = sampling_steps[0]
    s = sampling_steps[1]
    ni = b_spline_deg[0]
    ns = b_spline_deg[1]

    # TODO what is this nc?
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
        # TODO why the -theta?
        kernel = get_kernel_table(nt, ni, ns, h, s, -theta)

    if angledeg:
        theta = np.deg2rad(theta)

    spline_image = change_basis.change_basis(image, 'cardinal', 'b-spline', ni)
    sinogram = cspline.radon(
        spline_image,
        h,
        ni,
        center[1],
        center[0],
        # TODO why?
        -theta,
        kernel[0],
        kernel[1],
        nc,
        s,
        ns,
        captors_center
    )

    if ns > -1:
        sinogram = change_basis_columns.change_basis_columns(sinogram, 'dual', 'cardinal', ns)

    return sinogram


def spliradon(sinogram, theta=None, angledeg=True, n=None, filter_type='RAM-LAK',
              b_spline_deg=(-1, 1), sampling_steps=(1, 1),
              center=None, captors_center=None, kernel=None):
    nc = sinogram.shape[0]
    na = sinogram.shape[1]

    ni = b_spline_deg[0]
    ns = b_spline_deg[1]
    h = sampling_steps[0]
    s = sampling_steps[1]

    if theta is None:
        theta = np.pi / na
    theta = np.atleast_1d(theta)

    if theta.size == 1 and na > 1:
        theta = np.arange(na) * theta

    if len(theta) != na:
        raise ValueError("Theta does not match the number of projections in the provided sinogram.")

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
        # TODO again
        kernel = get_kernel_table(nt, ni, ns, h, s, -theta)

    sinogram, pre_filter = filter_sinogram.filter_sinogram(sinogram, filter_type, ns)

    if pre_filter:
        sinogram = change_basis_columns.change_basis_columns(sinogram, 'CARDINAL', 'B-SPLINE', ns)

    if angledeg:
        theta = np.deg2rad(theta)

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

    if ni > -1:
        image = change_basis.change_basis(image, 'DUAL', 'CARDINAL', ni)

    if theta.size > 1:
        image = image * np.pi / theta.size

    return image
