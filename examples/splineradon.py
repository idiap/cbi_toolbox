import cbi_toolbox as cbi
import cbi_toolbox.splineradon
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

image = np.zeros((70, 70, 3))
image[10, 3, 0] = 1
image[10, 50, 0] = 1
image[50, 50, 1] = 1
image[60, 20, 1] = 1
image[30, 60, 2] = 1
image[30, 50, 2] = 1

use_cuda = cbi.splineradon.is_cuda_available()
if use_cuda:
    print('Running with GPU acceleration')
else:
    print('Running on CPU')

sinogram = cbi.splineradon.splradon(image, use_cuda=use_cuda)

reconstruct = cbi.splineradon.spliradon(sinogram, use_cuda=use_cuda)

reconstruct -= reconstruct.min()
reconstruct /= reconstruct.max()

sinogram -= sinogram.min()
sinogram /= sinogram.max()

plt.figure()
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(sinogram)
plt.subplot(133)
plt.imshow(reconstruct)
plt.savefig('splineradon_example.pdf')
