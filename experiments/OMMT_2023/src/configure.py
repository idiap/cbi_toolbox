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
    for var_name, var_values in variables:
        conf = base_conf.copy()

        for val in var_values:
            conf[var_name] = val

            case_dir = outdir.joinpath(var_name, str(val))
            case_dir.mkdir(exist_ok=True, parents=True)

            with (case_dir / f"config.json").open("w") as fp:
                json.dump(conf, fp)


def gen_lsurface(base_conf, x_var, y_var, outdir):
    x_name, x_var = x_var
    for x_val in x_var:
        conf = base_conf.copy()
        conf[x_name] = x_val
        gen_config(conf, [y_var], outdir / str(x_val))


if __name__ == "__main__":
    default_conf = {
        # Imaging parameters
        "sample": "line_5",
        "photons": 10000,
        "dyn_range": 0.9,
        "min_intensity": 0,
        "n_mod": 16,
        "n_max": 32,
        "modulation": "hadamard",
        # Reconstruction parameters
        "target_res": 128,
        "NA": 0.5,
        "model": "1D",
        "niter": 50,
        "denoise": "TV2",
        "rho": 1,
        "sigma": 0.15,
        "lamZ": 0.0001,
        "lamXY": 1e-2,
        "rtol": 0.01,
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str)

    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)

    lambda_xy = ("lamXY", (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1))
    lambda_z = ("lamZ", (0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))

    variables = [("sigma", [0.05, 0.1, 0.2, 0.3, 0.6, 0.9])]
    l1_variables = [("lamZ", (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1))]

    samples = (
        "grid",
        "multiline",
        "moon",
    )

    for sample in samples:
        exp = outdir / f"TV_{sample}"
        default_conf["sample"] = sample
        gen_lsurface(default_conf, lambda_z, lambda_xy, exp)

    default_conf["denoise"] = "L1"
    for sample in samples:
        exp = outdir / f"L1_{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, l1_variables, exp)

    default_conf["denoise"] = "BM4D"
    for sample in samples:
        exp = outdir / f"BM_{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, variables, exp)

    default_conf["denoise"] = "TV"
    default_conf["sample"] = "multiline"
    exp = outdir / "TV1_multiline"
    gen_config(default_conf, l1_variables, exp)

    samples = (
        "grid",
        "multiline",
    )

    default_conf["photons"] = 400

    default_conf["denoise"] = "TV2"
    for sample in samples:
        exp = outdir / f"nTV_{sample}"
        default_conf["sample"] = sample
        gen_lsurface(default_conf, lambda_z, lambda_xy, exp)

    default_conf["denoise"] = "BM4D"
    for sample in samples:
        exp = outdir / f"nBM_{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, variables, exp)

    default_conf["denoise"] = "L1"
    for sample in samples:
        exp = outdir / f"nL1_{sample}"
        default_conf["sample"] = sample
        gen_config(default_conf, l1_variables, exp)

    default_conf["denoise"] = "TV"
    default_conf["sample"] = "multiline"
    exp = outdir / "nTV1_multiline"
    gen_config(default_conf, l1_variables, exp)
