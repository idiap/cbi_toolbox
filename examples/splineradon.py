import cbi_toolbox.splineradon as srad
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


image = np.zeros((70, 70, 3))
image[10, 3, 0] = 1
image[10, 50, 0] = 1
image[50, 50, 1] = 1
image[60, 20, 1] = 1
image[30, 60, 2] = 1
image[30, 50, 2] = 1

plt.figure()
plt.imshow(image)
plt.show()

sinogram = srad.splradon(image)

reconstruct = srad.spliradon(sinogram)

reconstruct -= reconstruct.min()
reconstruct /= reconstruct.max()

plt.figure()
plt.imshow(reconstruct)
plt.show()