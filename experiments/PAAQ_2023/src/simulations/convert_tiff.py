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

import tifffile
import pathlib
import numpy as np
import scipy.spatial.distance as scidist
import scipy.ndimage as scind
import skimage.transform
import json
import argparse

if __name__ == "__main__":
    default_path = pathlib.Path(__file__).absolute().parent.parent / "output"

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--outdir", type=str, default=str(default_path))
    parser.add_argument("--name", type=str, default="hs_10x")
    parser.add_argument("--downsample", type=float, default=2)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir) / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.data_path)

    files = sorted(data_path.glob("*.tif"))

    images = [tifffile.imread(file) for file in files]

    sequence = np.stack(images)

    down_sequence = skimage.transform.rescale(
        sequence, 1 / args.downsample, channel_axis=0
    )

    initial_phase = 0
    dist = scidist.cdist(
        down_sequence[initial_phase].reshape(1, -1),
        down_sequence.reshape(down_sequence.shape[0], -1),
        metric="minkowski",
        p=1,
    )[0]

    dist /= dist.max()

    dist = scind.gaussian_filter1d(dist, 0.8)

    diff = dist[1:] - dist[:-1]

    mini = np.logical_and(
        np.sign(diff[1:]) * np.sign(diff[:-1]) == -1, np.sign(diff[1:]) == 1
    )

    mini = np.logical_and(mini, dist[1:-1] < 0.55)

    period_starts = np.where(mini)[0] + 1
    period_starts = np.concatenate(([0], period_starts))

    periods = period_starts[1:] - period_starts[:-1]

    sequence_params = {
        "periods": periods.tolist(),
        "period_starts": period_starts[:-1].tolist(),
        "framerate": 1000,
    }

    with (outdir / "{}.json".format(args.name)).open("w") as fp:
        json.dump(sequence_params, fp)

    np.save(outdir / args.name, sequence)
