import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cbi_toolbox.splineradon as spl
matplotlib.use('Agg')

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

plt.savefig(os.path.join(os.path.dirname(__file__), 'splineradon_example.pdf'))
