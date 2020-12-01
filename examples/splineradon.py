# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>

# This file is part of CBI Toolbox.

# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import cbi_toolbox.splineradon as spl

image = np.zeros((71, 71, 3))
image[2, 2, 0] = 1
image[50, 50, 0] = 1
image[10, 50, 1] = 1
image[60, 20, 1] = 1
image[30, 60, 2] = 1
image[30, 20, 2] = 1
image[35, 35, :] = 1

use_cuda = spl.is_cuda_available()
if use_cuda:
    print('Running with GPU acceleration')
else:
    print('Running on CPU')

sinogram = spl.radon(image, use_cuda=use_cuda)

reconstruct = spl.iradon(sinogram, use_cuda=use_cuda)

reconstruct -= reconstruct.min()
reconstruct /= reconstruct.max()

sinogram[sinogram < 0] = 0
sinogram /= sinogram.max()

plt.figure()
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(sinogram)
plt.subplot(133)
plt.imshow(reconstruct)

plt.show()
