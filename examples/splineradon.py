import cbi_toolbox as cbi
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((70, 70, 3))
image[10, 3, 0] = 1
image[10, 50, 0] = 1
image[50, 50, 1] = 1
image[60, 20, 1] = 1
image[30, 60, 2] = 1
image[30, 50, 2] = 1

sinogram = cbi.splradon(image)

reconstruct = cbi.spliradon(sinogram)

reconstruct -= reconstruct.min()
reconstruct /= reconstruct.max()

plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(reconstruct)
plt.show()
