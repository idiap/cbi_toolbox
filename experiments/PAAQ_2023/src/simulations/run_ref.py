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

import numpy as np
import scipy.spatial.distance as scidist
import pathlib
import argparse
import json
import skimage.transform


def est_phase(ref_period, samples, downsampling):
    ref_period = skimage.transform.rescale(ref_period, 1 / downsampling, channel_axis=0)
    samples = skimage.transform.rescale(samples, 1 / downsampling, channel_axis=0)

    dist = scidist.cdist(
        ref_period.reshape(ref_period.shape[0], -1),
        samples.reshape(samples.shape[0], -1),
        metric="minkowski",
        p=1,
    )

    ref_phases = np.argmin(dist, 0) / ref_period.shape[0]

    return ref_phases


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("conf_file", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", type=int, default=-1)

    args = parser.parse_args()

    config_file = pathlib.Path(args.conf_file).absolute()

    with config_file.open("r") as fp:
        config = json.load(fp)

    output_path = config_file.parent.resolve()

    data_file = pathlib.Path(config["data_file"])
    data_config = data_file.with_suffix(".json")

    ref_data = np.load(str(data_file))
    with data_config.open("r") as fp:
        ref_config = json.load(fp)

    periods = ref_config["periods"]
    period_starts = np.array(ref_config["period_starts"])
    framerate = ref_config["framerate"]

    seed = args.seed

    if args.run_id >= 0:
        output_path = output_path / f"{args.run_id:02d}"
        output_path.mkdir(exist_ok=True)
        seed = args.run_id

    downsampling = config["ref_downsampling"]
    n_frames = config["n_frames"]
    f_camera = config["f_camera"]
    frame_gap = config["frame_gap"]
    n_channels = config["channels"]
    ref_step = config["ref_step"]

    sample_step = int(framerate // (f_camera / 2))
    gap_step = int(frame_gap * framerate)

    span = int(sample_step * n_frames)
    max_start = ref_data.shape[0] - span

    ref_arr = []
    sig_arr = []
    phase_arr = []

    rng = np.random.default_rng(seed)

    for idx_channel in range(n_channels):
        start_idx = rng.integers(0, max_start)
        period_idx = np.argmax(start_idx < period_starts) - 1

        refs = ref_data[start_idx : start_idx + span : sample_step]
        sigs = ref_data[
            start_idx + gap_step : start_idx + gap_step + span : sample_step
        ]

        ref_start = period_starts[period_idx]
        ref_period = periods[period_idx]
        ref_sequence = ref_data[ref_start : ref_start + ref_period : ref_step]
        phases = est_phase(ref_sequence, sigs, downsampling)

        f_signal = framerate / ref_period

        ref_arr.append(refs)
        sig_arr.append(sigs)
        phase_arr.append(phases)

    ref_arr = np.stack(ref_arr)
    sig_arr = np.stack(sig_arr)
    phase_arr = np.stack(phase_arr)

    path = output_path / "heart_reference.npz"

    np.savez_compressed(
        path, reference=ref_arr, signal=sig_arr, phase=phase_arr, f_signal=f_signal
    )
