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
import logging

import scico.numpy as snp
from scico import linop, loss, functional
from scico.optimize import admm

import numpy as np
import skimage.transform
import cbi_toolbox.reconstruct as cbir

root = pathlib.Path(__file__).parent
sys.path.append(str(root))
from tools import compute_sampling_line


class ScanImaging(linop.LinearOperator):
    def __init__(self, sampling):
        self.sampling = sampling
        input_shape = (sampling.shape[1], sampling.shape[2])
        out_shape = (sampling.shape[0],)

        super().__init__(
            input_shape=input_shape,
            output_shape=out_shape,
            input_dtype=self.sampling.dtype,
            jit=True,
        )

    def _eval(self, x):
        return snp.einsum("td,ntd->n", x, self.sampling)

    def _adj(self, y):
        return snp.einsum("n,ntd->td", y, self.sampling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    path = pathlib.Path(args.measure)
    measures = np.load(path)
    path = path.parent

    config_file = None

    if args.config is not None:
        config_file = pathlib.Path(args.config)
        path = config_file.parent

    if args.outdir is not None:
        path = pathlib.Path(args.outdir)

    logging.basicConfig(filename=str(path / "log.txt"), level=logging.INFO)

    if config_file is None:
        config_file = path / "config.json"
        if not config_file.exists():
            config_file = path.parent / "config.json"

    with config_file.open("r") as fp:
        config = json.load(fp)

    try:
        data_path = config["data_path"]
        data_path = pathlib.Path(data_path)
    except KeyError:
        data_path = path.parent.parent / "data"
        if not data_path.exists():
            data_path = data_path.parent.parent / "data"
        if not data_path.exists():
            data_path = data_path.parent.parent / "data"
        if not data_path.exists():
            data_path = data_path.parent.parent / "data"
        if not data_path.exists():
            exc = FileNotFoundError(f"Data path does not exist: {str(data_path)}")
            logging.exception(exc)
            raise exc

    logging.info(f"Using config:\n{config}")

    if (path / "results.npz").exists():
        logging.info("Results exist, skipping")
        exit(0)

    target_res = config["target_res"]
    target_framerate = config["target_framerate"]
    sampling_slope = config["slope"]

    rho = config["rho"]
    denoise = config["denoise"]
    niter = config["niter"]
    lamZ_list = config["lamZ_list"]
    lamT_list = config["lamT_list"]
    rtol = config["rtol"]

    psf = np.load(data_path / "psf.npy")

    measure = measures["measure"]
    phases = measures["phases"]
    modulation = measures["modulation"]

    sample = config["sample"]

    target_shape = (target_framerate, target_res)

    sample = np.load(data_path / f"{sample}.npy")
    downsize = (sample.shape[0] // target_shape[0], sample.shape[1] // target_shape[1])
    sample = skimage.transform.downscale_local_mean(sample, downsize)

    start_i = int((len(psf) // 2) % downsize[1])
    psf = psf[start_i :: downsize[1]]

    measure = measure / measure.max()

    pm_zips = zip(phases, modulation)

    measurement_matrix = np.empty((modulation.shape[0], target_framerate, target_res))
    for index, pm_zip in enumerate(pm_zips):
        phase, modu = pm_zip
        measurement_matrix[index] = compute_sampling_line(
            target_framerate, target_res, sampling_slope, phase, modu, psf
        )

    modu_mat = snp.asarray(measurement_matrix)
    measure = snp.asarray(measure)

    sampling = ScanImaging(modu_mat)
    scaling = (sampling.T @ measure).max()
    sampling = ScanImaging(modu_mat / scaling)

    f = loss.SquaredL2Loss(y=measure, A=sampling)

    if denoise == "TV":
        dT = linop.FiniteDifference(input_shape=target_shape, axes=0, circular=True)
        dZ = linop.FiniteDifference(input_shape=target_shape, axes=1)

        g1 = functional.L1Norm()
        g2 = functional.L1Norm()

        C = [dT, dZ]

    elif denoise == "L1":
        g1 = functional.L1Norm()

        C = [linop.Identity(target_shape)]

        # Empty list of lambdas for unused regu
        lamZ_list = [-1]

    else:
        raise NotImplementedError()

    rho_list = [rho] * len(C)

    x0 = sampling.T @ measure

    # Alternative initial guess
    # x0 = snp.mean(x0, 1)
    # x0 = snp.repeat(x0[..., None], target_shape[1], axis=1)

    x0 = snp.full(target_shape, snp.mean(x0), measure.dtype)

    def early_stop(opti):
        var = snp.abs(opti.x - opti.old_x).max() / opti.old_x.max()
        variations.append(var)

        if var < rtol:
            raise Exception("Early convergence")

        opti.old_x = opti.x

    if rtol <= 0:
        early_stop = None

    results_outer = {
        "time": [],
        "converged": [],
        "psnr": [],
        "regT": [],
        "regZ": [],
        "cost": [],
        "solution": [],
    }

    for lamT in lamT_list:
        results_inner = {}

        for key in results_outer:
            results_inner[key] = []

        for lamZ in lamZ_list:
            logging.info(f"Using lambdas: ({lamT}, {lamZ})")

            if denoise == "TV":
                g = [lamT * g1, lamZ * g2]

            elif denoise == "L1":
                g = [lamT * g1]

            variations = []
            solver = admm.ADMM(
                f=f,
                g_list=g,
                C_list=C,
                rho_list=rho_list,
                x0=x0,
                maxiter=niter,
                subproblem_solver=admm.LinearSubproblemSolver(
                    cg_kwargs={"tol": 1e-6, "maxiter": 100}
                ),
                itstat_options={"display": True},
            )
            solver.old_x = solver.x

            converged = False

            try:
                sol = solver.solve(callback=early_stop)
            except Exception as e:
                logging.info(str(e))
                solver.timer.stop()
                solver.itnum += 1
                solver.itstat_object.end()
                sol = solver.x
                converged = True

            itst = solver.itstat_object
            hist = itst.history()
            header = (
                (" " * itst.colsep).join(
                    [
                        "%-*s" % (fl, fn)
                        for fl, fn in zip(itst.fieldlength, itst.fieldname)
                    ]
                )
                + "\n"
                + "-" * itst.headlength
            )

            log = ["", header]
            for item in hist:
                values = tuple(item)
                log.append((" " * itst.colsep).join(itst.fieldformat) % values)

            log = "\n".join(log)
            logging.info(log)

            if len(variations) > 0:
                vari = "\n".join(
                    [f"{it} \t {var}" for (it, var) in enumerate(variations)]
                )
                logging.info("Variations")
                logging.info(vari)

            sol = np.clip(sol, 0, None)

            cost = float(f(sol))

            psnr = cbir.psnr(sample, sol, norm="mse")

            regT = float(g1(C[0] @ sol))
            if denoise == "TV":
                regZ = float(g2(C[1] @ sol))
            else:
                logging.info("No Z regu")
                regZ = 0

            results_inner["time"].append(solver.timer.elapsed())
            results_inner["converged"].append(converged)
            results_inner["psnr"].append(psnr)
            results_inner["regT"].append(regT)
            results_inner["regZ"].append(regZ)
            results_inner["cost"].append(cost)
            results_inner["solution"].append(sol)

        for key, val in results_inner.items():
            results_outer[key].append(val)

    out_file = path / "results.npz"
    np.savez(out_file, **results_outer)
