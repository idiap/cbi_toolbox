"""
This package implements radon and inverse radon transforms using b-spline interpolation formulas.
"""

import numpy as np

from cbi_toolbox.splineradon import filter_sinogram, spline_kernels, steps

try:
    from cbi_toolbox.cudaradon import is_cuda_available as _cuda_available
except ImportError:
    def _cuda_available():
        raise ModuleNotFoundError(
            "cbi_toolbox.cudaradon not found, did you install with CUDA?\n"
            "Try verbose install to spot errors (pip install -v).")


def is_cuda_available(verbose=False):
    try:
        return _cuda_available()
    except Exception as e:
        if verbose:
            print(e)
        return False


def radon(image, theta=np.arange(180), angledeg=True, n=None,
          b_spline_deg=(1, 3), sampling_steps=(1, 1),
          center=None, captors_center=None, kernel=None, use_cuda=False):
    """
    Perform a radon transform on the image.

    :param image: [z, x, y]
    :param theta:
    :param angledeg:
    :param n:
    :param b_spline_deg:
    :param sampling_steps:
    :param center:
    :param captors_center:
    :param kernel:
    :param use_cuda:
    :return: sinogram [angles, captors, y]
    """

    spline_image = steps.radon_pre(image, b_spline_deg)
    sinogram = steps.radon_inner(spline_image, theta, angledeg, n, b_spline_deg, sampling_steps,
                                 center, captors_center, kernel, use_cuda=use_cuda)

    sinogram = steps.radon_post(sinogram, b_spline_deg)

    return sinogram


def iradon(sinogram, theta=None, angledeg=True, filter_type='RAM-LAK',
           b_spline_deg=(1, 2), sampling_steps=(1, 1),
           center=None, captors_center=None, kernel=None, use_cuda=False):
    """
    Perform an inverse radon transform (backprojection) on the sinogram.

    :param sinogram: [angles, captors, y]
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
    :return: image [z, x, y]
    """

    sinogram = steps.iradon_pre(sinogram, b_spline_deg, filter_type)

    image, theta = steps.iradon_inner(sinogram, theta, angledeg, b_spline_deg, sampling_steps,
                                      center, captors_center, kernel, use_cuda=use_cuda)
    image = steps.iradon_post(image, theta, b_spline_deg)

    return image
