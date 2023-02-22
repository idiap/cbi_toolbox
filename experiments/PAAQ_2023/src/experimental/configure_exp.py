# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
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

import pathlib
import numpy as np
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)
    parser.add_argument("--downsampling", type=int, default=2)
    parser.add_argument("--ksize", type=int, default=50)
    parser.add_argument("--lpoints", type=int, default=11)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--npoints", type=int, default=-1)
    parser.add_argument("--wwidth", type=float, default=0.3)

    args = parser.parse_args()

    src_path = pathlib.Path(args.data)

    data = np.load(str(src_path / "data_cropped.npz"))

    if args.npoints < 0:
        N_list = [20, 50, 100]

    else:
        N_list = [args.npoints]

    for N in N_list:
        data_path = src_path / str(N)
        print(f"Creating output path {str(data_path)}")
        data_path.mkdir(parents=True, exist_ok=True)

        print(f"Keeping {N} points")
        refs = data["reference"][:, :N].copy()
        sigs = data["signal"][:, :N].copy()

        print("Saving truncated data")
        np.savez(str(data_path / "data"), reference=refs, signal=sigs)

        parameters = {
            "f_camera": 1 / 70e-3,
            "downsampling": args.downsampling,
            "w_width": args.wwidth,
            "k_size": args.ksize,
            "l_points": args.lpoints,
            "passes": args.passes,
            "subsampling": 4,
            "bins": 20,
        }

        print("Saving parameters")
        with (data_path / "config.json").open("w") as fp:
            json.dump(parameters, fp)

    print("Data preparation done")
