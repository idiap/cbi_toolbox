"""
The fpsopt module implements algorithms to deconvolve sinograms acquired using
Focal-Plane-Scanning OPT [1].

**Conventions:**

arrays follow the ZXY convention, with

    - Z : depth axis (axial, focus axis)
    - X : horizontal axis (lateral)
    - Y : vertical axis (lateral, rotation axis when relevant)

sinograms follow the TPY convention, with

    - T : angles (theta)
    - P : captor axis
    - Y : rotation axis

[1] K. G. Chan and M. Liebling, *"Direct inversion algorithm for focal plane
scanning optical projection tomography"*, in Biomedical Optics Express, vol. 8,
no. 11, pp. 5349-5358, 2017.
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


# This code is derived from `fpsopt`, which includes the following license:
#
#  Modified BSD-2 License - for Non-Commercial Research and Educational Use Only
#
# Copyright (c) 2017, The Regents of the University of California
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for non-commercial research and educational use
# only provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# For permission to use for commercial purposes, please contact UCSB's Office of
# Technology & Industry Alliances at 805-893-5180 or info@tia.ucsb.edu.

from scipy import fft
import numpy as np
from cbi_toolbox.utils import fft_size


def inverse_psf_rfft(psf, shape=None, l=20, mode='laplacian'):
    """
    Computes the real FFT of a regularized inversed 2D PSF (or projected 3D)
    This follows the convention of fft.rfft: only half the spectrum is computed.

    Parameters
    ----------
    psf : array [ZXY] or [XY]
        The 2D PSF (if 3D, will be projected on Z axis).
    shape : tuple (int, int), optional
        Shape of the full-sized desired PSF
        (if None, will be the same as the PSF), by default None.
    l : int, optional
        Regularization lambda, by default 20
    mode : str, optional
        The regularizer used, by default laplacian.
        One of: ['laplacian', 'constant']

    Returns
    -------
    array [XY]
        The real FFT of the inverse PSF.

    Raises
    ------
    ValueError
        If the PSF has incorrect number of dimensions.
        If the regularizer is unknown.
    """

    if psf.ndim == 3:
        psf = psf.sum(0)
    elif psf.ndim != 2:
        raise ValueError("Invalid dimensions for PSF: {}".format(psf.ndim))

    if shape is None:
        shape = psf.shape

    psf_fft = fft.rfft2(psf, s=shape)

    # We need to shift the PSF so that the center is located at the (0, 0) pixel
    # otherwise deconvolving will shift every pixel
    freq = fft.rfftfreq(shape[1])
    phase_shift = freq * 2 * np.pi * ((psf.shape[1] - 1) // 2)
    psf_fft *= np.exp(1j * phase_shift[None, :])

    freq = fft.fftfreq(shape[0])
    phase_shift = freq * 2 * np.pi * ((psf.shape[0] - 1) // 2)
    psf_fft *= np.exp(1j * phase_shift[:, None])

    if mode == 'laplacian':
        # Laplacian regularization filter to avoid NaN
        filt = [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]
        reg = np.abs(fft.rfft2(filt, s=shape)) ** 2
    elif mode == 'constant':
        reg = 1
    else:
        raise ValueError('Unknown regularizer: {}'.format(mode))

    return psf_fft.conjugate() / (np.abs(psf_fft) ** 2 + l * reg)


def deconvolve_sinogram(sinogram, psf, l=20, mode='laplacian', clip=True):
    """
    Deconvolve a sinogram with given PSF.

    Parameters
    ----------
    sinogram : numpy.ndarray [TPY]
        The blurred sinogram.
    psf : numpy.ndarray [ZXY] or [XY]
        The PSF used to deconvolve.
    l : float, optional
        Strength of the regularization.
    clip : bool, optional
        Clip negative values to 0, default is True.

    Returns
    -------
    numpy.ndarray [TPY]
        The deconvolved sinogram.
    """

    fft_shape = [fft_size(s) for s in sinogram.shape[1:]]

    inverse = inverse_psf_rfft(psf, shape=fft_shape, l=l, mode=mode)

    s_fft = fft.rfft2(sinogram, s=fft_shape)
    i_fft = fft.irfft2(s_fft * inverse, s=fft_shape, overwrite_x=True)
    i_fft = i_fft[:, :sinogram.shape[1], :sinogram.shape[2]]
    if clip:
        np.clip(i_fft, 0, None, out=i_fft)

    return i_fft


if __name__ == '__main__':
    from cbi_toolbox.simu import optics, primitives, imaging
    import cbi_toolbox.splineradon as spl
    import napari

    TEST_SIZE = 64

    s_psf = optics.gaussian_psf(
        numerical_aperture=0.3,
        npix_axial=TEST_SIZE+1, npix_lateral=TEST_SIZE+1)

    i_psf = inverse_psf_rfft(s_psf, l=1e-15, mode='constant')
    psfft = fft.rfft2(s_psf.sum(0))
    dirac = fft.irfft2(psfft * i_psf, s=s_psf.shape[1:])

    sample = primitives.boccia(
        TEST_SIZE, radius=(0.8 * TEST_SIZE) // 2, n_stripes=4)
    s_theta = np.arange(90)

    s_radon = spl.radon(sample, theta=s_theta, circle=True)
    s_fpsopt = imaging.fps_opt(sample, s_psf, theta=s_theta)

    s_deconv = deconvolve_sinogram(s_fpsopt, s_psf, l=0)

    viewer = napari.view_image(s_radon)
    viewer.add_image(s_fpsopt)
    viewer.add_image(s_deconv)

    viewer = napari.view_image(fft.fftshift(
        np.abs(i_psf), 0), name='inverse PSF FFT')
    viewer.add_image(dirac)

    napari.run()
