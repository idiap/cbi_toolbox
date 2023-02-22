"""
This example reproduces the figures shown on the main page of the documentation.
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

import numpy as np
import napari

from cbi_toolbox.simu import primitives, optics, textures
from cbi_toolbox.simu import imaging
from cbi_toolbox import splineradon as spl
from cbi_toolbox.reconstruct import fpsopt

TEST_SIZE = 128
TEST_SHAPE = (TEST_SIZE,) * 3

CMAP = "magma"
ATTEN = 0.03
ROTATE = (-20, 20, 0)

s_texture = (textures.simplex(TEST_SHAPE, seed=0, scale=12) + 1) / 2
s_texture += 0.3
s_texture[s_texture > 1] = 1

sample = primitives.torus_boccia(
    TEST_SIZE,
    radius=(0.8 * TEST_SIZE) // 2,
    n_stripes=1,
    deg_space=20,
    torus_radius=0.09,
)

t_sample = sample * s_texture

s_psf = optics.gaussian_psf(
    numerical_aperture=0.5,
    npix_axial=TEST_SIZE + 1,
    npix_lateral=TEST_SIZE + 1,
    pixelscale=3e-7,
)

s_theta = np.arange(180)

s_fpsopt = imaging.fps_opt(t_sample, s_psf, theta=s_theta)

s_deconv = fpsopt.deconvolve_sinogram(s_fpsopt, s_psf)

r_deconv = spl.iradon(s_deconv, s_theta, circle=True)

viewer = napari.view_image(
    sample,
    rendering="attenuated_mip",
    attenuation=ATTEN,
    colormap=CMAP,
    rotate=ROTATE,
    ndisplay=3,
)
viewer.add_image(
    s_texture,
    rendering="attenuated_mip",
    attenuation=ATTEN,
    colormap=CMAP,
    rotate=ROTATE,
)
viewer.add_image(
    t_sample,
    rendering="attenuated_mip",
    attenuation=ATTEN,
    colormap=CMAP,
    rotate=ROTATE,
)
viewer.add_image(
    r_deconv,
    rendering="attenuated_mip",
    attenuation=ATTEN,
    colormap=CMAP,
    rotate=ROTATE,
)

napari.run()

viewer2 = napari.view_image(np.log10(s_psf + 1e-12), colormap=CMAP)

napari.run()

viewer3 = napari.view_image(s_fpsopt, colormap=CMAP)
viewer3.add_image(s_deconv, colormap=CMAP)

napari.run()
