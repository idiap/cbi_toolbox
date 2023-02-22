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

from cbi_toolbox.simu import dynamic, primitives, textures, imaging
import numpy as np
import pathlib
import argparse
import json


def heart_simu(
    phases,
    size=128,
    ellipse_center=(0.5, 0.5),
    ellipse_radius=(0.4, 0.36),
    ellipse_thickness=0.15,
    ellipse_smoothing=0.5,
    texture_scale=25,
    texture_min=0.2,
    texture_seed=0,
):

    transformed = dynamic.sigsin_beat(phases, size, dtype=np.float32)

    ellipse = primitives.forward_ellipse(
        transformed,
        ellipse_center,
        ellipse_radius,
        ellipse_thickness,
        ellipse_smoothing,
    )

    texture = np.empty_like(ellipse)

    for index, layer_coords in enumerate(transformed):
        textures.forward_simplex(
            layer_coords,
            scale=texture_scale,
            out=texture[index, ...],
            seed=texture_seed,
        )

    tex_scale = (1 - texture_min) / 2
    texture = tex_scale * texture + tex_scale + texture_min

    ellipse *= texture

    return transformed, ellipse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("conf_file", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", type=int, default=-1)

    args = parser.parse_args()

    config_file = pathlib.Path(args.conf_file).absolute()

    with config_file.open("r") as fp:
        config = json.load(fp)

    output_path = config_file.parent

    seed = args.seed

    if args.run_id >= 0:
        output_path = output_path / f"{args.run_id:02d}"
        output_path.mkdir(exist_ok=True)
        seed = args.run_id

    size = config["size"]
    n_frames = config["n_frames"]

    mask_shape = config["mask_shape"]
    scale = config["scale"]

    f_signal = config["f_signal"]
    f_camera = config["f_camera"]

    sigma = config["sigma"]

    photons = config["photons"]

    rng = np.random.default_rng(seed)

    # ellipse shape and texture
    mask_args = []
    for mask in mask_shape:
        center, radius = mask
        radius = np.array(radius) * scale
        mask_args.append((center, radius, 0, 10, True))

    sig_arr = []
    phase_arr = []

    phases_ = []
    for ind_channel in range(len(mask_args) + 1):
        phases_.append(
            dynamic.sample_phases(
                n_frames,
                f_signal,
                f_camera,
                sigma,
                seed=seed + ind_channel,
            )
        )
    phases_ = np.concatenate(phases_)

    coordinates_, images_ = heart_simu(phases_, size=size, texture_seed=seed)

    for mask_idx, mask in enumerate(mask_args):

        coordinates = coordinates_[mask_idx * n_frames : (mask_idx + 1) * n_frames]
        signals = images_[mask_idx * n_frames : (mask_idx + 1) * n_frames]
        phases = phases_[mask_idx * n_frames : (mask_idx + 1) * n_frames]

        mask_ellipse = primitives.forward_ellipse(coordinates, *mask)

        signals = signals * mask_ellipse

        signals = imaging.quantize(
            imaging.noise(signals, photons=photons, max_amp=1, seed=seed, in_place=True)
        )

        sig_arr.append(signals)
        phase_arr.append(phases)

    refs = images_[-n_frames:]
    ref_phases = phases_[-n_frames:]

    refs = imaging.quantize(
        imaging.noise(refs, photons=photons, max_amp=1, seed=seed, in_place=True)
    )

    sig_arr.append(refs)
    phase_arr.append(ref_phases)

    sig_arr = np.stack(sig_arr)
    phase_arr = np.stack(phase_arr)

    path = output_path / "heart_mutual.npz"

    np.savez_compressed(path, signal=sig_arr, phase=phase_arr)
