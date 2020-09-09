import math
import astropy.units as u
import numpy as np
import poppy
import scipy.interpolate


def create_wf_1d(wf_object, upsampling=1, scale=1, copy=False):
    if copy:
        wf_object = wf_object.copy()

    wf = np.ones(
        (1, int(wf_object.shape[1] * upsampling)), dtype=wf_object.wavefront.dtype)
    y, x = np.indices(wf.shape, dtype=float)
    x -= wf.shape[1] / 2

    wf_object._x = x
    wf_object._y = y
    wf_object.wavefront = wf
    wf_object.pixelscale = wf_object.pixelscale / upsampling * scale
    wf_object.n = wf.shape[1]

    return wf_object


def wf_to_2d(wf_object, npix=None, copy=False):
    if copy:
        wf_object = wf_object.copy()

    if npix is None:
        size = wf_object.shape[1]
    else:
        size = npix
    center = wf_object.shape[1] // 2
    hw = size // 2

    new_wf = np.zeros_like(wf_object.wavefront, shape=(size, size))

    new_wf[hw, :] = wf_object.wavefront[:, center - hw:center + hw]
    wf_object.wavefront = new_wf

    wf_object._y, wf_object._x = np.indices(wf_object.shape, dtype=float)
    wf_object._y -= wf_object.shape[0] / 2.0
    wf_object._x -= wf_object.shape[0] / 2.0

    return wf_object


def wf_mix(wf1, wf2, ref=None):
    wfa = wf1.wavefront.squeeze()
    wfb = wf2.wavefront.squeeze()

    mix = np.outer(wfb, wfa)

    if ref is None:
        wf_m = wf1.copy()
    else:
        wf_m = ref.copy()

    wf_m.wavefront = mix

    return wf_m


def resample_wavefront(wf, pixelscale, npixels):
    pixscale_ratio = (wf.pixelscale / pixelscale).decompose().value

    def make_axis(npix, step):
        """ Helper function to make coordinate axis for interpolation """
        return step * np.arange(-npix // 2, npix // 2, dtype=np.float64)

    # Input and output axes for interpolation.  The interpolated wavefront will be evaluated
    # directly onto the detector axis, so don't need to crop afterwards.
    x_in = make_axis(wf.shape[1], wf.pixelscale.to(u.m / u.pix).value)
    x_out = make_axis(npixels.value, pixelscale.to(u.m / u.pix).value)

    def interpolator(arr):
        """
        Bind arguments to scipy's RectBivariateSpline function.
        For data on a regular 2D grid, RectBivariateSpline is more efficient than interp2d.
        """
        return scipy.interpolate.interp1d(
            x_in, arr, kind='slinear', copy=False, fill_value=0,
            assume_sorted=True, bounds_error=False)

    # Interpolate real and imaginary parts separately
    real_resampled = interpolator(wf.wavefront.real)(x_out)
    imag_resampled = interpolator(wf.wavefront.imag)(x_out)
    new_wf = real_resampled + 1j * imag_resampled

    # enforce conservation of energy:
    new_wf *= 1. / pixscale_ratio

    wf.ispadded = False  # if a pupil detector, avoid auto-cropping padded pixels on output
    wf.wavefront = new_wf
    wf.pixelscale = pixelscale


def spim_illumination(wavelength=500e-9, refr_index=1.333, laser_radius=1.2e-3,
                      objective_na=0.3, objective_focal=18e-3,
                      slit_opening=10e-3, fov_width=1.3e-3,
                      npix_fov=512, simu_size=2048, oversample=16):

    pixel_width = 1
    wavelength *= u.m
    laser_radius *= u.m
    objective_focal *= u.m
    fov_width *= u.m
    slit_opening *= u.m

    noop = poppy.ScalarTransmission()
    beam_ratio = 1 / oversample

    fov_pixels = npix_fov * u.pixel
    fov_pixelscale = fov_width / fov_pixels
    detector = poppy.FresnelOpticalSystem()
    detector.add_detector(fov_pixels=fov_pixels, pixelscale=fov_pixelscale)

    # We approximate the objective aperture with a square one to make it separable
    # Given the shape of the wavefront, we estimate the generated error to be negligible
    objective_radius = math.tan(math.asin(objective_na)) * objective_focal
    objective_aperture = poppy.RectangleAperture(name='objective aperture',
                                                 width=2 * objective_radius,
                                                 height=2 * objective_radius)
    objective_lens = poppy.QuadraticLens(
        f_lens=objective_focal, name='objective lens')

    obj_aperture = poppy.FresnelOpticalSystem()
    obj_aperture.add_optic(objective_aperture, objective_focal)

    # Implement the objective lens separately to be able to account for refractive index change
    obj_lens = poppy.FresnelOpticalSystem()
    obj_lens.add_optic(objective_lens)

    # Computed as following: going through T1 then CLens then T2
    # is equivalent to going through CLens with focal/4
    # Then the radius is computed as the Fourier transform of the input beam, per 2F lens system
    w0_y = (12.5e-3 * u.m * wavelength) / (2 * np.pi ** 2 * laser_radius)
    laser_shape_y = poppy.GaussianAperture(w=w0_y, pupil_diam=5 * w0_y)
    path_y = poppy.FresnelOpticalSystem(
        pupil_diameter=2 * w0_y, npix=pixel_width, beam_ratio=beam_ratio)
    path_y.add_optic(laser_shape_y)

    laser_shape_z = poppy.GaussianAperture(
        w=laser_radius, pupil_diam=slit_opening / 2)
    slit = poppy.RectangleAperture(
        name='Slit', width=slit_opening / 2, height=slit_opening / 2)

    # Going through T1, slit and T2 is equivalent to going through a half-sized slit,
    # then propagating 1/4 the distance
    # Since we use 1D propagation, we can increase oversampling a lot for better results
    path_z = poppy.FresnelOpticalSystem(
        pupil_diameter=slit_opening / 2, npix=pixel_width, beam_ratio=beam_ratio)
    path_z.add_optic(laser_shape_z)
    path_z.add_optic(slit)
    path_z.add_optic(noop, 0.25 * 100e-3 * u.m)

    # Propagate 1D signals
    wf_z = path_z.input_wavefront(wavelength=wavelength)
    create_wf_1d(wf_z, upsampling=simu_size)
    path_z.propagate(wf_z)

    wf_y = path_y.input_wavefront(wavelength=wavelength)
    create_wf_1d(wf_y, upsampling=simu_size, scale=10)
    path_y.propagate(wf_y)

    obj_aperture.propagate(wf_z)
    obj_aperture.propagate(wf_y)

    wf_z.wavelength /= refr_index
    wf_y.wavelength /= refr_index

    obj_lens.propagate(wf_z)
    obj_lens.propagate(wf_y)

    illumination = np.empty(
        (npix_fov, npix_fov, npix_fov), dtype=wf_z.intensity.dtype)

    for pix in range(npix_fov):
        pixel = pix - npix_fov // 2
        distance = pixel * (fov_width / npix_fov)

        psf = poppy.FresnelOpticalSystem()
        psf.add_optic(noop, objective_focal + distance)

        wfc_y = wf_y.copy()
        wfc_z = wf_z.copy()

        psf.propagate(wfc_y)
        psf.propagate(wfc_z)

        resample_wavefront(wfc_y, fov_pixelscale, fov_pixels)
        resample_wavefront(wfc_z, fov_pixelscale, fov_pixels)

        mix = wf_mix(wfc_y, wfc_z)
        mix.normalize()

        illumination[:, pix, :] = mix.intensity

    return illumination


if __name__ == '__main__':
    import napari

    illu = spim_illumination(
        simu_size=1024, npix_fov=256, oversample=8)

    with napari.gui_qt():
        napari.view_image(illu)
