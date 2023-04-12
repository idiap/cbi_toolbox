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


def gen_config(base_conf, variables, outdir, device):
    for var_name, var_values in variables:
        conf = base_conf.copy()

        for val in var_values:
            conf[var_name] = val

            case_dir = outdir.joinpath(var_name, str(val))
            case_dir.mkdir(exist_ok=True, parents=True)

            with (case_dir / f"config_{device}.json").open("w") as fp:
                json.dump(conf, fp)


def gen_lsurface(base_conf, x_var, y_var, outdir, device):
    x_name, x_var = x_var
    for x_val in x_var:
        conf = base_conf.copy()
        conf[x_name] = x_val
        gen_config(conf, [y_var], outdir / str(x_val), device)


if __name__ == "__main__":
    default_conf = {
        "width": 128,
        "compression": 8,
        "niter": 2,
        "denoise": "TV2",
        "warmup": 2,
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str)

    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)

    width = ("width", (128, 256, 512, 1024))
    compress = ("compression", (8, 4, 2))

    out_ = outdir / f"t_TV"
    default_conf["denoise"] = "TV2"
    gen_lsurface(default_conf, width, compress, out_, "cpu")
    out_ = outdir / f"t_TV_cuda"
    gen_lsurface(default_conf, width, compress, out_, "cuda")

    out_ = outdir / f"t_TV1"
    default_conf["denoise"] = "TV"
    gen_lsurface(default_conf, width, compress, out_, "cpu")
    out_ = outdir / f"t_TV1_cuda"
    gen_lsurface(default_conf, width, compress, out_, "cuda")

    out_ = outdir / f"t_L1"
    default_conf["denoise"] = "L1"
    gen_lsurface(default_conf, width, compress, out_, "cpu")
    out_ = outdir / f"t_L1_cuda"
    gen_lsurface(default_conf, width, compress, out_, "cuda")

    out_ = outdir / f"t_BM"
    default_conf["denoise"] = "BM4D"
    gen_lsurface(default_conf, width, compress, out_, "cpu")
