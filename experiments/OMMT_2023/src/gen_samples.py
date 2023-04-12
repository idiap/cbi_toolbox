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

import numpy as np
import argparse
import pathlib
from cbi_toolbox.simu import primitives, optics, imaging
from cbi_toolbox.utils import ome

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("--antialias", type=int, default=4)
    parser.add_argument("--psf_size", type=int, default=127)
    parser.add_argument("--scale", type=float, default=0.8)
    args = parser.parse_args()

    outpath = pathlib.Path(args.outdir) / "data"

    n_psf = args.psf_size
    antialias = args.antialias

    radius = int(args.scale * 128 / 2)

    try:
        psf, _ = ome.load_ome_tiff(outpath / "bw_psf.ome.tiff")
        print(f"Using provided PSF of size {psf.shape}")
    except FileNotFoundError:
        print("Using Gaussian model PSF")
        psf = optics.gaussian_psf(
            npix_axial=n_psf, npix_lateral=n_psf, wavelength=600e-9
        )

    spim = optics.openspim_illumination(npix_fov=128, npix_z=127, slit_opening=4e-3)

    multiline = primitives.fiber(
        128, (30, 80, 64), (0.1, -0.1, 1), 2, "circle", antialias=antialias
    )
    multiline += (
        primitives.fiber(
            128, (70, 15, 64), (-0.7, 0.2, 1), 4, "diamond", antialias=antialias
        )
        * 0.6
    )
    multiline += (
        primitives.fiber(
            128, (45, 64, 64), (0.4, 1, 1), 3, "square", antialias=antialias
        )
        * 0.8
    )
    multiline += (
        primitives.fiber(
            128, (80, 70, 64), (-0.6, -0.4, 1), 3, "circle", antialias=antialias
        )
        * 0.7
    )

    x0 = 20
    step = 22
    grid = primitives.fiber(128, (x0, x0, x0), (0, 0, 1), 2, antialias=antialias)
    for i in range(1, 5):
        x = x0 + i * step
        grid += primitives.fiber(128, (x, x, x), (0, 0, 1), 2, antialias=antialias)

    moon = primitives.fiber(128, (64, 64, 64), (0, 0, 1), 30, antialias=antialias)
    small = primitives.fiber(128, (64, 80, 64), (0, 0, 1), 20, antialias=antialias)
    moon -= small
    moon = np.clip(moon, 0, None)

    def im_save(sample, name):
        image = imaging.spim(sample, psf, spim)
        np.save(outpath / name, sample)
        np.save(outpath / f"im_{name}", image)

    im_save(multiline, "multiline")
    im_save(grid, "grid")
    im_save(moon, "moon")

    np.save(outpath / "psf", psf)
    np.save(outpath / "spim", spim)
