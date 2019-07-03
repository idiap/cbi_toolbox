"""
This package implements radon and inverse radon transforms using b-spline interpolation formulas.
"""

import numpy as np

from cbi_toolbox.splineradon import filter_sinogram, spline_kernels, steps


def splradon(image, theta=np.arange(180), angledeg=True, n=None,
             b_spline_deg=(1, 3), sampling_steps=(1, 1),
             center=None, captors_center=None, kernel=None):
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
    :return:
    """

    spline_image = steps.splradon_pre(image, b_spline_deg)

    sinogram = steps.splradon_inner(spline_image, theta, angledeg, n, b_spline_deg, sampling_steps,
                                    center, captors_center, kernel)

    sinogram = steps.splradon_post(sinogram, b_spline_deg)

    return sinogram


def spliradon(sinogram, theta=None, angledeg=True, filter_type='RAM-LAK',
              b_spline_deg=(1, 2), sampling_steps=(1, 1),
              center=None, captors_center=None, kernel=None):
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
    :return:
    """

    sinogram = steps.spliradon_pre(sinogram, b_spline_deg, filter_type)

    image, theta = steps.spliradon_inner(sinogram, theta, angledeg, b_spline_deg, sampling_steps,
                                         center, captors_center, kernel)

    image = steps.spliradon_post(image, theta, b_spline_deg)

    return image
