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
import time
import cbi_toolbox.splineradon as spl

width = 200
depth = 500

image = np.random.default_rng().standard_normal((width, width, depth))

use_cuda = spl.is_cuda_available()
if use_cuda:
    print("Running with GPU acceleration")
else:
    print("Running on CPU")

start = time.time()
sinogram = spl.radon(image, use_cuda=use_cuda)
stop = time.time()

print("Radon : {}".format(stop - start))

del image

start = time.time()
reconstruct = spl.iradon(sinogram, use_cuda=use_cuda)
stop = time.time()

print("IRadon : {}".format(stop - start))
