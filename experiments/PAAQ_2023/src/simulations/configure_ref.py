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

import argparse
import pathlib
import json


def gen_config(base_conf, variables, outdir):
    for var_name, var_values in variables:

        conf = base_conf.copy()

        for val in var_values:
            conf[var_name] = val

            case_dir = outdir.joinpath(var_name, str(val))
            case_dir.mkdir(exist_ok=True, parents=True)

            with (case_dir / "config.json").open("w") as fp:
                json.dump(conf, fp)


if __name__ == "__main__":
    default_conf = {
        "n_frames": 50,
        "frame_gap": 5e-3,
        "f_camera": 20,
        "data_name": "hs_10x",
        "channels": 2,
        "ref_step": 2,
        "ref_downsampling": 1,
        # Reconstruction parameters
        "downsampling": 1,
        "w_width": 3e-1,
        "k_size": 50,
        "l_points": 21,
        "passes": 2,
    }

    default_path = pathlib.Path(__file__).absolute().parent.parent / "output"

    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default=str(default_path))

    args = parser.parse_args()

    data_dir = pathlib.Path(args.outdir) / "data"
    default_conf["data_file"] = str(
        data_dir / "{}.npy".format(default_conf["data_name"])
    )

    outdir = pathlib.Path(args.outdir) / "ref"

    outdir.mkdir(parents=True, exist_ok=True)

    variables = [
        ("n_frames", (15, 25, 50, 75, 100)),
    ]

    gen_config(default_conf, variables, outdir)
