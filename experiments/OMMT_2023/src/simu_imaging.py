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

import numpy as np
import scipy.linalg as scilin
from cbi_toolbox.simu import imaging
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    path = pathlib.Path(args.config)

    with path.open("r") as fp:
        config = json.load(fp)

    path = path.parent


    seed = args.seed

    if seed is None:
        seed = 0
    else:
        path = path / f"{seed:02d}"
        path.mkdir(exist_ok=True)

    logging.basicConfig(filename=str(path / "im_log.txt"), level=logging.INFO)

    try:
        data_path = config["data_path"]
        data_path = pathlib.Path(data_path)
    except KeyError:
        data_path = path.parent.parent.parent / "data"
        if not data_path.exists():
            data_path = data_path.parent.parent / "data"
        if not data_path.exists():
            data_path = data_path.parent.parent / "data"
        if not data_path.exists():
            exc = FileNotFoundError(f"Data path does not exist: {str(data_path)}")
            logging.exception(exc)
            raise exc

    logging.info(f"Using config:\n{config}")

    if (path / "measure.npy").exists():
        logging.info("Images exist, skipping")
        exit(0)

    n_mod = config["n_mod"]
    n_max = config["n_max"]
    photons = config["photons"]
    dyn_range = config["dyn_range"]
    min_intensity = config["min_intensity"]
    modulation = config["modulation"]
    na = config["NA"]

    sample = np.load(data_path / f"{config['sample']}.npy")
    image = np.load(data_path / f"im_{config['sample']}.npy")
    spim = np.load(data_path / "spim.npy")

    resolution = sample.shape[0]

    rng = np.random.default_rng(seed)

    if modulation == "hadamard":
        hada_mat = scilin.hadamard(n_max, dtype=float)

        choice = rng.choice(n_max - 1, n_mod - 1, replace=False)
        choice = choice + 1
        choice = np.concatenate((choice, (0,)))
        choice = np.sort(choice)

        hada_mat = hada_mat[choice]
        modu_mat = (hada_mat + 1) / 2

    elif modulation == "uniform":
        ran_mat = rng.uniform(0, 1, (n_mod, n_max))
        ran_mat -= ran_mat.min(1, keepdims=True)
        ran_mat /= ran_mat.max(1, keepdims=True)
        ran_mat *= 1 - min_intensity
        ran_mat += min_intensity
        modu_mat = ran_mat

    else:
        raise ValueError(f"Incorrect modulation: {modulation}")

    illu_mat = np.repeat(modu_mat, resolution // n_max, -1)

    def simu_camera(image_in, dyn_range, photons, seed):
        image_in /= image_in.max()
        image_in *= dyn_range
        image_in = imaging.quantize(
            imaging.noise(image_in, photons=photons, seed=seed, clip=True, max_amp=1)
        )
        return image_in

    measure = np.tensordot(illu_mat, image, 1)
    measure = simu_camera(measure, dyn_range, photons, seed)

    ref_measure = simu_camera(image, dyn_range, photons, seed)

    out_file = path / "measure.npy"
    np.save(out_file, measure)

    out_file = path / "reference.npy"
    np.save(out_file, ref_measure)

    out_file = path / "illumination.npy"
    np.save(out_file, modu_mat)
