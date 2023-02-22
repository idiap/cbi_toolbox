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

import argparse
import pathlib
import json
import numpy as np
import skimage.transform
from cbi_toolbox.reconstruct import vhf, mutual_information
import scipy.signal as signal


def register_sequence(reference, moving, bins=20, subsampling=1):
    """
    Computes the frame difference between two periodic sequences using mutual
    information. The inputs must be sequences of same length covering a single
    period.

    Parameters
    ----------
    reference : array_like, [N, ...]
        The reference sequence of size N, covering one period of the signal
    moving : array_like, [N, ...]
        The sequence to register, of size N, covering one period of the signal
    bins : int, optional
        Number of bins for mutual informatino, by default 20
    subsampling : int, optional
        Subsampling factor allowing for sub-frame registration, by default 1

    Returns
    -------
    tuple (float, float)
        The positive shift (or roll) needed to align the moving sequence to the
        reference sequence, and the value of the mutual information for the
        obtained solution.
    """

    if subsampling > 1:
        reference = signal.resample(reference, reference.shape[0] * subsampling, axis=0)
        moving = signal.resample(moving, moving.shape[0] * subsampling, axis=0)

    mutual = []
    for shift in range(reference.shape[0]):
        temp = np.roll(moving, shift, axis=0)
        mut = mutual_information(reference, temp, bins)
        mutual.append(mut)

    argmax = np.argmax(mutual)
    return argmax / subsampling, mutual[argmax]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)

    args = parser.parse_args()

    path = pathlib.Path(args.file).absolute().parent

    config_file = path / "config.json"

    if not config_file.exists():
        config_file = path.parent / "config.json"

    with config_file.open("r") as fp:
        parameters = json.load(fp)

    print(f"Running Mutual Information on {args.file}")
    print(f"Using parameters {parameters}")

    data = np.load(args.file)

    images = data["signal"]

    downsampling = parameters["downsampling"]
    subsampling = parameters["subsampling"]
    bins = parameters["bins"]

    # Sort the channels
    seqs = []
    phases = []
    for seq in images:
        seq = skimage.transform.rescale(seq, 1 / downsampling, channel_axis=0)
        phi_naive = vhf.vhf_phase_naive(seq)
        sorting = np.argsort(phi_naive)

        phases.append(phi_naive)
        seqs.append(seq[sorting])

    phases = np.array(phases)

    for idx, seq in enumerate(seqs[:-1]):
        delta, corr = register_sequence(
            seqs[-1], seq, bins=bins, subsampling=subsampling
        )
        delta_b, corr_b = register_sequence(
            seqs[-1], seq[::-1], bins=bins, subsampling=subsampling
        )

        # Detect flipped solutions
        if corr_b > corr:
            phases[idx] = (-phases[idx]) % 1
            delta = delta_b - 1

        delta /= seqs[0].shape[0]
        phases[idx] += delta
        phases[idx] %= 1

    out_file = path / "mutual_solution"

    np.savez(str(out_file), phase=phases)

    try:
        gt_phase = data["phase"]
        gt_phase = gt_phase[:-1].reshape(-1)

        phases = phases[:-1].reshape(-1)

        err = min(
            vhf.phase_error(gt_phase, phases).mean(),
            vhf.phase_error(gt_phase, 1 - phases).mean(),
        )

        results = {
            "phase": err,
        }

        res_file = path / "results.json"

        with res_file.open("w") as fp:
            json.dump(results, fp)

    except:
        pass
