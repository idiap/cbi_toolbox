"""
This example demonstrates how to simulate a 3D beating heart.
"""

# Copyright (c) 2023 Idiap Research Institute, http://www.idiap.ch/
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

import napari
import numpy as np

from cbi_toolbox.simu import primitives, textures, dynamic

TEST_SIZE = 64
TIME_SIZE = 32

# Uniform sampling over one period
phases = np.linspace(0, 1, TIME_SIZE, endpoint=False)

# Basic contraction of the coordinates
coords = dynamic.sigsin_beat_3(phases, TEST_SIZE)
print("Done simulating contraction")

# Heart walls as an ellipsoid
ellipse = primitives.forward_ellipse_3(coords, center=(.5, .5, .5), radius=(.2, .3, .4))
print("Done simulating the heart wall")

# Adding texture using simplex noise
simplex = textures.forward_simplex(coords, scale=20, time=True, seed=0)
print("Done simulating the heart texture")

del coords
simplex += 2
simplex /= simplex.max()
heart = ellipse * simplex

viewer = napari.view_image(
    heart,
    rendering="attenuated_mip",
    attenuation=0.1,
    colormap="magma",
    ndisplay=3,
)

napari.run()
