import time
import numpy as np
import napari

from cbi_toolbox.simu import primitives, optics
from cbi_toolbox.simu import imaging
from cbi_toolbox import splineradon as spl
from cbi_toolbox.reconstruct import deconv

TEST_SIZE = 64

sample = primitives.boccia(
    TEST_SIZE, radius=(0.8 * TEST_SIZE) // 2, n_stripes=4)
s_psf = optics.gaussian_psf(
    numerical_aperture=0.3,
    npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)
opt_psf = optics.gaussian_psf(
    numerical_aperture=0.1, npix_axial=TEST_SIZE, npix_lateral=TEST_SIZE+1)
spim_illu = optics.openspim_illumination(
    slit_opening=2e-3,
    npix_fov=TEST_SIZE, simu_size=4*TEST_SIZE, rel_thresh=1e-3)
s_theta = np.arange(180)

start = time.time()
s_widefield = imaging.widefield(sample, s_psf)
print('Time for widefield: \t{}s'.format(time.time() - start))
start = time.time()
noisy = imaging.noise(s_widefield)
print('Time for noise: \t{}s'.format(time.time() - start))

start = time.time()
s_spim = imaging.spim(sample, s_psf, spim_illu)
print('Time for SPIM: \t{}s'.format(time.time() - start))

start = time.time()
s_opt = imaging.opt(sample, opt_psf, theta=s_theta)
print('Time for OPT: \t{}s'.format(time.time() - start))

start = time.time()
s_fpsopt = imaging.fps_opt(sample, s_psf, theta=s_theta)
print('Time for FPS-OPT: \t{}s'.format(time.time() - start))

start = time.time()
s_fssopt = imaging.fss_opt(sample, s_psf, spim_illu, theta=s_theta)
print('Time for FSS-OPT: \t{}s'.format(time.time() - start))

start = time.time()
s_radon = spl.radon(sample, theta=s_theta, pad=False)
print('Time for radon: \t{}s'.format(time.time() - start))

start = time.time()
s_deconv = deconv.deconvolve_sinogram(s_fpsopt, s_psf)
print('Time for deconv: \t{}s'.format(time.time() - start))


start = time.time()
r_radon = spl.iradon(s_radon, s_theta, unpad=False)
print('Time for reconstruct radon: \t{}s'.format(time.time() - start))

start = time.time()
r_opt = spl.iradon(s_opt, s_theta, unpad=False)
print('Time for reconstruct OPT: \t{}s'.format(time.time() - start))

start = time.time()
r_fpsopt = spl.iradon(s_fpsopt, s_theta, unpad=False)
print('Time for reconstruct FPS-OPT: \t{}s'.format(time.time() - start))

start = time.time()
r_deconv = spl.iradon(s_deconv, s_theta, unpad=False)
print('Time for reconstruct FPS-OPT deconv: \t{}s'.format(time.time() - start))

start = time.time()
r_fssopt = spl.iradon(s_fssopt, s_theta, unpad=False)
print('Time for reconstruct FSS-OPT: \t{}s'.format(time.time() - start))


with napari.gui_qt():
    viewer = napari.view_image(
        sample, rendering='attenuated_mip', attenuation=0.2, colormap='plasma')
    viewer.add_image(s_widefield, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(noisy, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(s_spim, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(r_radon, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(r_opt, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(r_fpsopt, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(r_deconv, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')
    viewer.add_image(r_fssopt, rendering='attenuated_mip',
                     attenuation=0.2, colormap='plasma')

    viewer = napari.view_image(s_radon, colormap='plasma')
    viewer.add_image(s_opt, colormap='plasma')
    viewer.add_image(s_fpsopt, colormap='plasma')
    viewer.add_image(s_deconv, colormap='plasma')
    viewer.add_image(s_fssopt, colormap='plasma')
