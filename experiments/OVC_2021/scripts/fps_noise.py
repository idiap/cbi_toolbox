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

import os
import numpy as np
import argparse

from cbi_toolbox.utils import ome
from cbi_toolbox.reconstruct import deconv
from cbi_toolbox.simu import optics, imaging
import cbi_toolbox.splineradon as spl

margin = False
photons = 1e4

path = os.environ["OVC_PATH"]

if margin:
    path = os.path.join(path, "margin")

ipath = os.path.join(path, "imaging")
npath = os.path.join(path, "noise")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)

args = parser.parse_args()
id = args.id - 1

na = id // 19
dna = id % 19

l = (1e4, 1e3, 1e2)[na]
na = (30, 50, 80)[na]
dna = 10 + 5 * dna

theta = np.linspace(0, 180, 360, endpoint=False)

fps_opt = np.load(os.path.join(ipath, "fpsopt_{:03d}.npy".format(na)))
fps_opt = imaging.noise(fps_opt, in_place=True, seed=0, photons=photons)

psf, _ = ome.load_ome_tiff(os.path.join(path, "psf", "BW_{:03d}.tif".format(dna)))
fpsf = optics.gaussian_psf(psf.shape[1], psf.shape[0], numerical_aperture=dna / 100)


deco = deconv.deconvolve_sinogram(fps_opt, psf, l=l)
np.save(os.path.join(npath, "{:03d}_{:03d}d.npy".format(na, dna)), deco)
deco = spl.iradon(deco, theta=theta, circle=True)
np.save(os.path.join(npath, "{:03d}_{:03d}.npy".format(na, dna)), deco)
print("saved {:03d}_{:03d}.npy".format(na, dna))

deco = deconv.deconvolve_sinogram(fps_opt, fpsf, l=l)
np.save(os.path.join(npath, "{:03d}_{:03d}fd.npy".format(na, dna)), deco)
deco = spl.iradon(deco, theta=theta, circle=True)
np.save(os.path.join(npath, "{:03d}_{:03d}f.npy".format(na, dna)), deco)
print("saved {:03d}_{:03d}f.npy".format(na, dna))
