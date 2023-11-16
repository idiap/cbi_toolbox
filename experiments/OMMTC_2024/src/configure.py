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
import json


def gen_config(base_conf, variables, outdir):
    for var_name in variables:
        conf = base_conf.copy()

        for val in variables[var_name]:
            conf[var_name] = val

            case_dir = outdir.joinpath(var_name, str(val))
            case_dir.mkdir(exist_ok=True, parents=True)

            with (case_dir / f"config.json").open("w") as fp:
                json.dump(conf, fp)


if __name__ == "__main__":
    default_conf = {
        # Imaging parameters
        "sample": "hollow",
        "photons": 10000,
        "theta_noise": 0,
        "dyn_range": 0.9,
        "min_intensity": 0,
        "n_samples": 800,
        "modulation": "hadamard",
        "n_mod": 32,
        "slope": 0.4,
        # Reconstruction parameters
        "target_res": 64,
        "target_framerate": 50,
        "niter": 500,
        "denoise": "TV",
        "rho": 1,
        "lamT_list": (
            1e-4,
            1e-3,
            1e-2,
            1e-1,
        ),  # If using L1, this is used
        "lamZ_list": (1e-4, 1e-3, 1e-2, 1e-1),  # If using L1, this is ignored
        "rtol": 5e-4,
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str)

    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)

    samples = (
        "hollow",
        # "solid",
    )

    variables = {
        "n_samples": [100, 200, 400, 800, 1600, 3200],
        "photons": [400, 1000, 10000],
        "slope": [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
        "theta_noise": [0, 1 / 200, 1 / 100, 1 / 50, 1 / 25],
    }

    for sample in samples:
        exp = outdir / f"{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, variables, exp)

    default_conf["n_samples"] = 400
    del variables["n_samples"]

    for sample in samples:
        exp = outdir / f"c8_{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, variables, exp)
