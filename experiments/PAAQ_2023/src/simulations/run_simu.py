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

    f_signal = config["f_signal"]
    f_camera = config["f_camera"]

    frame_sep = 1 / f_camera  # => f_acq = 1 / 50e-3 = 20Hz
    ref_exposure = config["ref_exposure"]
    frame_gap = config["frame_gap"]

    assert (ref_exposure + frame_gap) <= frame_sep, "Reference exposure is too long"

    first_pulse = frame_sep - frame_gap - ref_exposure

    pulse_amps = (1 / 6, 1 / 3, 1 / 2)
    pulse_sep = ref_exposure / 2
    frame_pattern = (
        f_camera
        * np.array(
            (
                first_pulse,
                first_pulse + pulse_sep,
                first_pulse + 2 * pulse_sep,
                frame_sep,
            )
        )
        / 2
    )

    sigma = config["sigma"]

    photons = config["photons"]

    rng = np.random.default_rng(seed)

    # ellipse shape and texture
    mask_args = [(*mask, 0, 10, True) for mask in mask_shape]

    ref_arr = []
    sig_arr = []
    phase_arr = []

    phases_ = []
    for ind_channel in range(len(mask_args)):
        phases_.append(
            dynamic.sample_phases(
                n_frames,
                f_signal,
                f_camera / 2,
                sigma,
                pattern=frame_pattern,
                seed=seed + ind_channel,
            )
        )
    phases_ = np.concatenate(phases_)

    n_images = len(frame_pattern) * n_frames

    coordinates_, images_ = heart_simu(phases_, size=size, texture_seed=seed)

    for mask_idx, mask in enumerate(mask_args):

        coordinates = coordinates_[mask_idx * n_images : (mask_idx + 1) * n_images]
        images = images_[mask_idx * n_images : (mask_idx + 1) * n_images]
        phases = phases_[mask_idx * n_images : (mask_idx + 1) * n_images]

        references = np.zeros_like(images, shape=(n_frames, *images.shape[1:]))

        for idx, amp in enumerate(pulse_amps):
            references += amp * images[idx :: len(frame_pattern)]

        signals = images[len(frame_pattern) - 1 :: len(frame_pattern)]
        coordinates = coordinates[len(frame_pattern) - 1 :: len(frame_pattern)]
        phases = phases[len(frame_pattern) - 1 :: len(frame_pattern)]

        mask_ellipse = primitives.forward_ellipse(coordinates, *mask)

        signals = signals * mask_ellipse

        signals = imaging.quantize(
            imaging.noise(signals, photons=photons, max_amp=1, seed=seed, in_place=True)
        )
        references = imaging.quantize(
            imaging.noise(
                references, photons=photons, max_amp=1, seed=seed, in_place=True
            )
        )

        ref_arr.append(references)
        sig_arr.append(signals)
        phase_arr.append(phases)

    ref_arr = np.stack(ref_arr)
    sig_arr = np.stack(sig_arr)
    phase_arr = np.stack(phase_arr)

    path = output_path / "heart_simulation.npz"

    np.savez_compressed(
        path, reference=ref_arr, signal=sig_arr, phase=phase_arr, f_signal=f_signal
    )
