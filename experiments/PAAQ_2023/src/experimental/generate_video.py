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
import pathlib

from cbi_toolbox.reconstruct import vhf
import imageio

import argparse

green = np.array((24, 136, 5))
red = np.array((232, 102, 119))
white = np.ones(3)

COLORS = red, green, white
GAINS = (1.5, 1.2, 0.3)


def save_vid(name, channels, colors=COLORS, gains=GAINS, scale=True):
    cols = []
    for col in colors:
        col = np.array(col, dtype=float)
        col /= col.max()
        cols.append(col)

    compose = np.empty(shape=(*channels[0].shape, 3), dtype=float)
    for idx, channel in enumerate(channels):
        chan = channel[..., None] - channel.min()
        chan = (chan / chan.max()) * cols[idx] * gains[idx]
        chan = np.clip(chan, 0, 1)

        compose += chan

    compose = (np.clip(compose, 0, 1) * 255).astype(np.uint8)

    scale_w = 154
    scale_h = 5 * channels.shape[-2] // 256
    scale_p = 20 * channels.shape[-2] // 256

    if scale:
        compose[:, -(scale_p + scale_h) : -scale_p, scale_p : scale_p + scale_w] = 255

    file_name = f"{name}.avi"

    imageio.mimsave(file_name, compose, fps=compose.shape[0] // 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str)
    parser.add_argument("--noscale", action="store_false")

    args = parser.parse_args()

    path = pathlib.Path(args.path)

    data = np.load(str(path / "data.npz"))
    sol = np.load(str(path / "mcvhf_solution.npz"))
    m_sol = np.load(str(path / "mutual_solution.npz"))

    sigs = data["signal"]

    phases = sol["phase"]
    mutu = m_sol["phase"]

    n_frames = sigs.shape[1]

    for channel in mutu:
        vhf.orient_phases(channel, in_place=True)

    phases -= phases[0, 0]
    mutu -= mutu[0, 0]

    phases %= 1
    mutu %= 1

    v_mcvhf = []
    v_mutu = []
    v_naive = []

    for chan_idx in range(phases.shape[0]):
        v_mcvhf.append(vhf.sample_video(sigs[chan_idx], phases[chan_idx], n_frames))
        v_mutu.append(vhf.sample_video(sigs[chan_idx], mutu[chan_idx], n_frames))

    v_mcvhf = np.array(v_mcvhf)
    v_mutu = np.array(v_mutu)

    v_all = np.concatenate((v_mutu[..., ::-1], v_mcvhf[..., ::-1]), axis=-1)

    use_scale = args.noscale

    save_vid(str(path / "comparative"), v_all, scale=use_scale)
