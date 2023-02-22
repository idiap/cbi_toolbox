"""
The utils package provides various utility functions to work with files and
arrays.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from cbi_toolbox.utils._arrays import *


def fft_size(n):
    """
    Returns the smallest power of 2 above n, but no less than 64
    for efficient FFT computations.

    Parameters
    ----------
    n : int
        Size of the signal.
    """

    return max(64, int(2 ** np.ceil(np.log2(n))))
