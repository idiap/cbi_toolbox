"""
The optics module provides simulations of the optics of imaging systems for microscopy

**Conventions:**

arrays follow the ZXY convention, with

    - Z : depth axis (axial, focus axis)
    - X : horizontal axis (lateral)
    - Y : vertical axis (lateral, rotation axis when relevant)
"""

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

import math
import astropy.units as u
import numpy as np
import poppy
import scipy.interpolate
from cbi_toolbox.simu import primitives
from cbi_toolbox import utils


def create_wf_1d(wf_object, upsampling=1, scale=1, copy=False):
    """
    Create a 1D wavefront object from an existing wavefront

    Parameters
    ----------
    wf_object : poppy.FresnelWavefront
        the original wavefront
    upsampling : int, optional
        upsampling factor (does not change the field of view), by default 1
    scale : int, optional
        zoom factor (changes the field of view), by default 1
    copy : bool, optional
        return a new object, by default False

    Returns
    -------
    poppy.FresnelWavefront
        a 1D wavefront full of 1 with same properties as the input
    """

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
    """
    Convert a 1D wavefront to 2D (for plotting only)

    Parameters
    ----------
    wf_object : poppy.FresnelWavefront
        the 1D wavefront
    npix : int, optional
        crop to a size of npix, by default None
    copy : bool, optional
        return a new object, by default False

    Returns
    -------
    poppy.FresnelWavefront
        the 2D wavefront
    """

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
    """
    Compute a 2D wavefront by multiplying 2 1D wavefronts (for separable propagation)

    Parameters
    ----------
    wf1 : poppy.FresnelWavefront
        a 1D wavefront
    wf2 : poppy.FresnelWavefront
        a 1D wavefront
    ref : poppy.FresnelWavefront, optional
        reference wavefront for the parameters of the output, by default None (wf1 will be used)

    Returns
    -------
    poppy.FresnelWavefront
        the 2D mixed wavefront

    Raises
    ------
    ValueError
        if the input wavefronts have different pixelscales
    """

    if wf1.pixelscale != wf2.pixelscale:
        raise ValueError("The pixelscale of the input wavefronts must match")

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
    """
    Resample 1D wavefront to new pixelscale
    (adapted from poppy.poppy_core._resample_wavefront_pixelscale)

    Parameters
    ----------
    wf : poppy.FresnelWavefront
        a 1D wavefront
    pixelscale : astropy.units.[distance] / astropy.units.pixel
        target pixelscale
    npixels : int
        target size in pixels

    Returns
    -------
    poppy.FresnelWavefront
        resampled and resized 1D wavefront
    """

    pixscale_ratio = (wf.pixelscale / pixelscale).decompose().value

    def make_axis(npix, step):
        """ Helper function to make coordinate axis for interpolation """
        return step * (np.arange(-npix // 2, npix // 2, dtype=np.float64))

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


def openspim_illumination(wavelength=500e-9, refr_index=1.333, laser_radius=1.2e-3,
                          objective_na=0.3, objective_focal=18e-3, slit_opening=10e-3,
                          pixelscale=635e-9, npix_fov=512, rel_thresh=None,
                          simu_size=2048, oversample=16):
    """
    Compute the illumination function of an OpenSPIM device

    Parameters
    ----------
    wavelength : float, optional
        illumination wavelength in meters, by default 500e-9
    refr_index : float, optional
        imaging medium refraction index, by default 1.333
    laser_radius : float, optional
        source laser radius in meters, by default 1.2e-3
    objective_na : float, optional
        illumination objective NA, by default 0.3
    objective_focal : float, optional
        illumination objective focal length in meters, by default 18e-3
    slit_opening : float, optional
        vertical slit opening in meters, by default 10e-3
    pixelscale : float, optional
        target pixelscale in meters per pixel, by default 1.3e-3/2048
    npix_fov : int, optional
        target size in pixels, by default 512
    rel_thresh: float, optional
        relative threshold to crop the beam thickness
        if a full row is below this theshold, all rows after are removed
        will be computed as compared to the maximum pixel
    simu_size : int, optional
        size of the arrays used for simulation, by default 2048
    oversample : int, optional
        oversampling used for the simulation (must be increased sith simu_size), by default 16

    Returns
    -------
    array [ZXY]
        the illumination function
    """

    pixel_width = 1
    wavelength *= u.m
    laser_radius *= u.m
    objective_focal *= u.m
    pixelscale *= (u.m / u.pixel)
    slit_opening *= u.m

    noop = poppy.ScalarTransmission()
    beam_ratio = 1 / oversample

    fov_pixels = npix_fov * u.pixel
    detector = poppy.FresnelOpticalSystem()
    detector.add_detector(fov_pixels=fov_pixels, pixelscale=pixelscale)

    # We approximate the objective aperture with a square one to make it separable
    # Given the shape of the wavefront, we estimate the generated error to be negligible
    objective_radius = math.tan(
        math.asin(objective_na / refr_index)) * objective_focal
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

    # Going through T1, slit and T2 is equivalent to going through a half-sized slit,
    # then propagating 1/4 the distance
    # Since we use 1D propagation, we can increase oversampling a lot for better results
    laser_shape_z = poppy.GaussianAperture(
        w=laser_radius, pupil_diam=slit_opening / 2)
    slit = poppy.RectangleAperture(
        name='Slit', width=slit_opening / 2, height=slit_opening / 2)
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

    # Make sure it is centered even if pixels are odd or even
    offset = 0 if npix_fov % 2 else 0.5

    for pix in range(npix_fov):
        pixel = pix - npix_fov // 2 + offset
        distance = pixel * pixelscale * u.pixel

        psf = poppy.FresnelOpticalSystem()
        psf.add_optic(noop, objective_focal + distance)

        wfc_y = wf_y.copy()
        wfc_z = wf_z.copy()

        psf.propagate(wfc_y)
        psf.propagate(wfc_z)

        resample_wavefront(wfc_y, pixelscale, fov_pixels)
        resample_wavefront(wfc_z, pixelscale, fov_pixels)

        mix = wf_mix(wfc_y, wfc_z)
        mix.normalize()

        illumination[:, pix, :] = mix.intensity

    if rel_thresh is not None:
        illumination = utils.threshold_crop(
            illumination, rel_thresh, 0)

    return illumination / illumination.sum(0).mean()


def gaussian_psf(npix_lateral=129, npix_axial=129,
                 pixelscale=635e-9, wavelength=500e-9,
                 numerical_aperture=0.5, refraction_index=1.33):
    """
    Compute an approximate PSF model based on gaussian beam propagation

    Koskela, O., Montonen, T., Belay, B. et al. Gaussian Light Model in Brightfield
    Optical Projection Tomography. Sci Rep 9, 13934 (2019).
    https://bib-ezproxy.epfl.ch:5295/10.1038/s41598-019-50469-6

    Parameters
    ----------
    npix_lateral : int, optional
        number of pixels in the lateral direction, by default 129
    npix_axial : int, optional
        number of pixels in the axial direction, by default 129
    pixelscale : float, optional
        pixelscale in meters per pixel, by default 1.3e-3/2048
    wavelength : float, optional
        illumination wavelength in meters, by default 500e-9
    numerical_aperture : float, optional
        objective NA, by default 0.5
    refraction_index : float, optional
        imaging medium NA, by default 1.33

    Returns
    -------
    array [ZXY]
        the gaussian PSF

    """

    # compensate for even/odd pixels so that the PSF is always centered
    odd_l = npix_lateral % 2
    odd_a = npix_axial % 2
    lat_offset = 0 if odd_l else 0.5
    ax_offset = 0 if odd_a % 2 else 0.5

    r_coords = (np.arange((npix_lateral + 1) // 2) + lat_offset) * pixelscale
    z_coords = (np.arange((npix_axial + 1) // 2) + ax_offset) * pixelscale

    w0 = wavelength / (np.pi * refraction_index * numerical_aperture)

    z_rayleygh = math.pi * w0 ** 2 * refraction_index / wavelength
    w_z = w0 * np.sqrt(1 + (z_coords/z_rayleygh)**2)
    w_zi2 = 1 / np.square(w_z)
    r_coords = np.square(r_coords)
    intens = np.sqrt(w0**2 * w_zi2)

    gauss_psf = np.einsum('i, ij -> ij', intens,
                          np.exp(- 2 * np.outer(w_zi2, r_coords)))
    gauss_psf = np.einsum('ij, ik->ijk', gauss_psf, gauss_psf)
    gauss_psf = primitives.quadrant_to_volume(gauss_psf, (odd_a, odd_l, odd_l))

    return gauss_psf


if __name__ == '__main__':
    import napari

    s_psf = gaussian_psf(npix_lateral=129, npix_axial=129)
    s_psf = np.log10(s_psf+1e-12)
    illu = openspim_illumination(
        simu_size=1024, npix_fov=256, oversample=8, rel_thresh=1e-6)

    with napari.gui_qt():
        viewer = napari.view_image(s_psf)
        viewer.add_image(illu)
