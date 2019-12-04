import cbi_toolbox as cbi
import cbi_toolbox.splineradon
import numpy as np
import time

width = 200
depth = 500

image = np.random.randn(width, width, depth)

use_cuda = cbi.splineradon.is_cuda_available()
if use_cuda:
    print('Running with GPU acceleration')
else:
    print('Running on CPU')

start = time.time()
sinogram = cbi.splineradon.radon(image, use_cuda=use_cuda)
stop = time.time()

print('Radon : {}'.format(stop-start))

del image

start = time.time()
reconstruct = cbi.splineradon.iradon(sinogram, use_cuda=use_cuda)
stop = time.time()

print('IRadon : {}'.format(stop-start))



