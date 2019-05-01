# B-spline interpolation function for degree up to 7
# Christian Jaques, june 2016, Computational Bioimaging Group, Idiap
# Francois Marelli, may 2019, Computational Bioimaging Group, Idiap
# This code is a translation of Michael Liebling's matlab code,
# which was already largely based on a C-library written by Philippe
# Thevenaz, BIG, EPFL

import math
import numpy as np
from scipy import signal

from cbi_toolbox.bsplines.bspline import compute_bspline
from cbi_toolbox.arrays import make_broadcastable


def convert_to_samples(c, deg, boundary_condition='Mirror'):
    """
        Convert interpolation coefficients into samples
        In the input array, the signals are considered along the first dimension (1D computations)
    """
    n = c.shape[0]
    if n == 1:
        return c

    kerlen = int(2 * math.floor(deg / 2.) + 1)

    k = -math.floor(deg / 2.) + np.arange(kerlen)
    kernel = compute_bspline(deg, k)
    # add boundaries to signal, extend it
    extens = int(math.floor(deg / 2.))

    # different extensions based on boundary condition
    if boundary_condition.upper() == 'MIRROR':
        c = np.concatenate((c[extens:0:-1, ...], c, c[n - 2:n - extens - 2:-1, ...]))
    elif boundary_condition.upper() == 'PERIODIC':
        c = np.concatenate((c[-extens:, ...], c, c[0:extens, ...]))
    else:
        raise ValueError('Illegal boundary condition: {}'.format(boundary_condition.upper()))

    kernel = make_broadcastable(kernel, c)

    c = signal.convolve(c, kernel, 'valid')

    return c
