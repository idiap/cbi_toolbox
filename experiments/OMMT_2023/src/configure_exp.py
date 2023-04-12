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

import pathlib
import numpy as np
import scipy.interpolate as interpolate
import json
import argparse

import cbi_toolbox.utils.ome as ome


def gen_config(base_conf, variables, outdir):
    for var_name, var_values in variables:
        conf = base_conf.copy()

        for val in var_values:
            conf[var_name] = val

            case_dir = outdir.joinpath(var_name, str(val))
            case_dir.mkdir(exist_ok=True, parents=True)

            with (case_dir / f"config_exp.json").open("w") as fp:
                json.dump(conf, fp)


def gen_lsurface(base_conf, x_var, y_var, outdir):
    x_name, x_var = x_var
    for x_val in x_var:
        conf = base_conf.copy()
        conf[x_name] = x_val
        gen_config(conf, [y_var], outdir / str(x_val))


def process_data(sample, outdir, data_dir):
    out_path = pathlib.Path(outdir) / f"exp_{sample}"
    gen_lsurface(default_conf, lambda_z, lambda_xy, out_path)

    if (out_path / "measure_exp.npy").exists():
        print(f"Sample {sample} exists, skipping")
        return

    path = pathlib.Path(data_dir)

    measure_ = ome.load_ome_tiff(str(path / sample / "mssr" / "mssr.ome.tif"))[0]
    reference_ = ome.load_ome_tiff(str(path / sample / "spim" / "spim.ome.tif"))[0]
    illumination = np.load(path / "illumination.npy")

    if sample == "s1":
        starts = (650, 300)
        dims = (640, 640)
    elif sample == "s2":
        starts = (870, 500)
        dims = (256, 256)

    crop = np.s_[:, starts[0] : starts[0] + dims[0], starts[1] : starts[1] + dims[1]]

    target_res = default_conf["target_res"]

    measure = measure_[crop]
    reference = reference_[crop]

    factor = reference.shape[0] // measure.shape[0]
    downsampled = reference[: factor * measure.shape[0] : factor, ...]

    x = np.arange(downsampled.shape[0]) * factor

    interp_f = interpolate.interp1d(
        x,
        downsampled,
        kind="cubic",
        axis=0,
        copy=False,
        fill_value=0,
        bounds_error=False,
    )

    downsampled = interp_f(np.arange(target_res))

    out_path.mkdir(exist_ok=True, parents=True)
    np.save(out_path / "measure_exp.npy", measure)
    np.save(out_path / "downsampled.npy", downsampled)
    np.save(out_path / "reference.npy", reference)
    np.save(out_path / "illumination.npy", illumination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str)
    parser.add_argument("data_dir", type=str)

    args = parser.parse_args()

    default_conf = {
        # Reconstruction parameters
        "target_res": 128,
        "NA": 0.5,
        "model": "None",
        "niter": 50,
        "denoise": "TV2",
        "rho": 1,
        "sigma": 0.15,
        "lamZ": 0.0001,
        "lamXY": 1e-2,
        "rtol": 0.01,
    }

    lambda_xy = ("lamXY", (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1))
    lambda_z = ("lamZ", (0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))

    for sample in ("s1", "s2"):
        process_data(sample, args.outdir, args.data_dir)
