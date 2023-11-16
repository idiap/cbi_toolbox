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
import sys

import numpy as np
import scipy.linalg as scilin
import logging
from cbi_toolbox.simu import imaging

root = pathlib.Path(__file__).parent
sys.path.append(str(root))
from tools import compute_sampling_line

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
    n_samples = config["n_samples"]
    photons = config["photons"]
    theta_noise = config["theta_noise"]
    dyn_range = config["dyn_range"]
    min_intensity = config["min_intensity"]
    modulation = config["modulation"]
    slope = config["slope"]

    sample = np.load(data_path / f"{config['sample']}.npy")
    psf = np.load(data_path / "psf.npy")

    n_time, n_depth = sample.shape

    rng = np.random.default_rng(seed)

    if modulation == "hadamard":
        hada_mat = scilin.hadamard(n_mod, dtype=float)
        modu_mat = (hada_mat + 1) / 2

    else:
        raise ValueError(f"Incorrect modulation: {modulation}")

    phases = rng.uniform(0, 1, n_samples)
    m_indices = rng.choice(n_mod, n_samples)
    phases_indices_zip = zip(phases, m_indices)

    measurement_matrix = np.empty((n_samples, n_time, n_depth))
    for index, pi_zip in enumerate(phases_indices_zip):
        phase, m_index = pi_zip
        measurement_matrix[index] = compute_sampling_line(
            n_time, n_depth, slope, phase, modu_mat[m_index], psf
        )

    def simu_camera(image_in, dyn_range, photons, seed):
        image_in /= image_in.max()
        image_in *= dyn_range
        image_in = imaging.quantize(
            imaging.noise(image_in, photons=photons, seed=seed, clip=True, max_amp=1)
        )
        return image_in

    measure = np.einsum("td,ntd->n", sample, measurement_matrix)
    measure = simu_camera(measure, dyn_range, photons, seed)

    phase_noise = rng.normal(0, theta_noise, size=phases.shape)
    phases += phase_noise
    phases %= 1

    np.savez(
        path / "measure.npz",
        measure=measure,
        phases=phases,
        modulation=modu_mat[m_indices],
    )
