# B-spline interpolation function for degree up to 7
# Christian Jaques, june 2016, Computational Bioimaging Group, Idiap
# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

from math import floor

from numpy import concatenate
from scipy.signal import convolve2d

from cbi_toolbox.splineradon.bspline import *


def convert_to_samples(c, deg, boundary_condition='Mirror'):
    n = c.shape[0]
    if n == 1:
        return c

    kerlen = int(2 * floor(deg / 2.) + 1)

    k = -floor(deg / 2.) + np.arange(kerlen)
    kernel = compute_bspline(deg, k)
    # add boundaries to signal, extend it
    extens = int(floor(deg / 2.))

    # different extensions based on boundary condition
    if boundary_condition == 'Mirror':
        c = concatenate((c[extens:0:-1], c, c[n - 2:n - extens - 2:-1]))
    elif boundary_condition == 'Periodic':
        c = concatenate((c[-extens::, :], c, c[0:extens, :]))

    # convolve2d needs arrays of 2d, even if one dimension is 1
    if len(c.shape) < 2:
        c = c[:, np.newaxis]
    if len(kernel.shape) < 2:
        kernel = kernel[:, np.newaxis]
    c = convolve2d(c, kernel, 'valid')

    return c
