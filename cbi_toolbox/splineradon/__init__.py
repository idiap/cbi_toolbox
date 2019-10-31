"""
This package implements radon and inverse radon transforms using b-spline interpolation formulas.
"""

import numpy as np

from cbi_toolbox.splineradon import filter_sinogram, spline_kernels, steps
# from cbi_toolbox.cudaradon import is_cuda_available as _cuda_available


# def is_cuda_available(verbose=False):
#     try:
#         return _cuda_available()
#     except Exception as e:
#         if verbose:
#             print(e)
#         return False


def splradon(image, theta=np.arange(180), angledeg=True, n=None,
             b_spline_deg=(1, 3), sampling_steps=(1, 1),
             center=None, captors_center=None, kernel=None, use_cuda=False):
    """
    Perform a radon transform on the image.

    :param image:
    :param theta:
    :param angledeg:
    :param n:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param kernel:
    :param use_cuda:
    :return:
    """

    spline_image = steps.splradon_pre(image, b_spline_deg)
    sinogram = steps.splradon_inner(spline_image, theta, angledeg, n, b_spline_deg, sampling_steps,
                                    center, captors_center, kernel, use_cuda=use_cuda)

    sinogram = steps.splradon_post(sinogram, b_spline_deg)

    return sinogram


def spliradon(sinogram, theta=None, angledeg=True, filter_type='RAM-LAK',
              b_spline_deg=(1, 2), sampling_steps=(1, 1),
              center=None, captors_center=None, kernel=None, use_cuda=False):
    """
    Perform an inverse radon transform (backprojection) on the sinogram.

    :param sinogram:
    :param theta:
    :param angledeg:
    :param n:
    :param filter_type:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param kernel:
    :param use_cuda:
    :return:
    """

    sinogram = steps.spliradon_pre(sinogram, b_spline_deg, filter_type)

    image, theta = steps.spliradon_inner(sinogram, theta, angledeg, b_spline_deg, sampling_steps,
                                         center, captors_center, kernel, use_cuda=use_cuda)
    image = steps.spliradon_post(image, theta, b_spline_deg)

    return image
