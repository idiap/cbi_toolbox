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
from cbi_toolbox.reconstruct import vhf

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

    f_acq_ref = parameters["f_camera"] / 2
    downsampling = parameters["downsampling"]
    w_width = parameters["w_width"]
    k_size = parameters["k_size"]
    n_points = parameters["l_points"]
    passes = parameters["passes"]

    print(f"Running MCVHF on {args.file}")
    print(f"Using parameters {parameters}")

    data = np.load(args.file)
    reference = data["reference"]
    seq_lengths = (reference.shape[1],) * reference.shape[0]

    references = reference.reshape((-1, *reference.shape[-2:]))
    references = skimage.transform.rescale(references, 1 / downsampling, channel_axis=0)

    phi_uni = vhf.vhf_phase_uncorrected(references, seq_lengths)

    try:
        opti_lambda = parameters["lambda"]
        print("Lambda given, skipping L-curve")
        losses = None

    except KeyError:
        losses, lam = vhf.l_curve(
            phi_uni,
            references,
            seq_lengths,
            w_width=w_width,
            k_size=k_size,
            n_points=n_points,
        )

        # Automatically select point in L-curve
        losses -= losses.min(1, keepdims=True)
        losses /= losses.max(1, keepdims=True)
        dist = (losses[0] + losses[1] * 10) ** 2
        amin = dist.argmin()
        opti_lambda = lam[amin]

    phi_est, status = vhf.phase_correction(
        phi_uni,
        references,
        seq_lengths,
        w_width,
        k_size,
        regu_lambda=opti_lambda,
        precision_passes=passes,
    )

    # total_iter = status[1]
    # d_loss, r_loss = status[2]

    f_est = vhf.average_frequency(phi_est, f_acq_ref)

    phi_naive = vhf.vhf_phase_naive(references)

    phi_est = phi_est.reshape(reference.shape[:2])
    phi_naive = phi_naive.reshape(reference.shape[:2])

    out_file = path / "mcvhf_solution"
    np.savez(str(out_file), frequency=f_est, phase=phi_est, naive=phi_naive)

    if losses is not None:
        lcurve_file = path / "lcurve"
        np.savez(str(lcurve_file), losses=losses, lambda_val=lam, amin=amin)

    try:
        gt_phase = data["phase"]
        gt_freq = data["f_signal"]

        gt_phase = gt_phase.reshape(-1)
        phi_est = phi_est.reshape(-1)
        phi_naive = phi_naive.reshape(-1)

        phi_err = vhf.phase_error(gt_phase, phi_est).mean()

        naive_err = min(
            vhf.phase_error(gt_phase, phi_naive).mean(),
            vhf.phase_error(gt_phase, 1 - phi_naive).mean(),
        )

        freq_err = np.abs(gt_freq - f_est) / gt_freq

        results = {
            "frequency": freq_err,
            "phase": phi_err,
            "naive": naive_err,
            "lambda": opti_lambda,
            # "iterations": total_iter,
            # "d_loss": d_loss,
            # "r_loss": r_loss,
        }

        res_file = path / "results.json"
        with res_file.open("w") as fp:
            json.dump(results, fp)

    except:
        pass
