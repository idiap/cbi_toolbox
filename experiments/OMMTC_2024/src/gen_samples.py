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

import argparse
import pathlib
import numpy as np
from cbi_toolbox.simu import dynamic, primitives, optics
import scipy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("--time", type=int, default=200)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--hadamard", type=int, default=32)
    args = parser.parse_args()

    outpath = pathlib.Path(args.outdir) / "data"

    print("Using Gaussian model PSF")
    psf = optics.gaussian_psf(
        npix_axial=127, npix_lateral=1, wavelength=600e-9
    ).squeeze()

    spim = optics.openspim_illumination(
        npix_fov=1, npix_z=127, slit_opening=4e-3, wavelength=600e-9
    ).squeeze()

    psf_z = psf * spim
    psf_z /= psf_z[63]

    phases = np.linspace(0, 1, args.time, endpoint=False)

    coords = dynamic.sigsin_beat(
        phases,
        args.depth,
        sigsin_slopes=(2, 2, 0),
        sigsin_saturations=(0.2, 0.2, 0),
        sigsin_init_phases=(0, 0, 0),
        sigsin_amplitudes=(2, 2, 0),
        wavelength=2,
        phase_0=0,
        beat_center=(0.5, 0.5),
        rotate_first=False,
        dtype=np.float64,
    )

    beat_hollow = primitives.forward_ellipse(
        coords, (0.5, 0.5), (0.4, 0.4), thickness=0.5, solid=False
    )
    hollow = beat_hollow[..., args.depth // 2]

    beat_solid = primitives.forward_ellipse(
        coords, (0.5, 0.5), (0.4, 0.4), thickness=0.5, solid=True
    )
    solid = beat_solid[..., args.depth // 2]

    outpath.mkdir(parents=True, exist_ok=True)
    np.save(outpath / "hollow.npy", hollow)
    np.save(outpath / "solid.npy", solid)
    np.save(outpath / "psf.npy", psf_z)
