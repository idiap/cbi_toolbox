import cbi_toolbox.splineradon as srad
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


image = np.zeros((50, 70))
image[10, 3] = 1

plt.figure()
plt.imshow(image)
plt.show()

sinogram = srad.splradon(image, n=1000, b_spline_deg=(1, 1))

plt.figure()
plt.imshow(np.reshape(sinogram,(sinogram.shape[1], sinogram.shape[0])))
plt.show()