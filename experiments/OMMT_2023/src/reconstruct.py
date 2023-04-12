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

import scico.numpy as snp
from scico import linop, loss, functional, denoiser
from scico.optimize import admm

import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import skimage.transform
from cbi_toolbox.simu import optics
import cbi_toolbox.reconstruct as cbir

import logging


class ScanImaging(linop.LinearOperator):
    def __init__(self, modulation, input_shape):
        self.modulation = modulation
        out_shape = (modulation.shape[0], *input_shape[-2:])

        super().__init__(
            input_shape=input_shape,
            output_shape=out_shape,
            input_dtype=self.modulation.dtype,
            jit=True,
        )

    def _eval(self, x):
        x = snp.tensordot(self.modulation, x, 1)
        return x

    def _adj(self, y):
        return snp.tensordot(self.modulation.T, y, 1)


class VDnCNN(functional.Functional):
    has_eval = False
    has_prox = True

    def __init__(self, variant="6N"):
        self.dncnn = denoiser.DnCNN(variant)
        if self.dncnn.is_blind:

            def denoise(x, sigma):
                return self.dncnn(x)

        else:

            def denoise(x, sigma):
                return self.dncnn(x, sigma)

        self._denoise = denoise

    def prox(self, v, lam=1.0, **kwargs):
        den_zx = self._denoise(v, lam)

        den_zy = self._denoise(v.transpose((0, 2, 1)), lam).transpose((0, 2, 1))
        den_xy = self._denoise(v.transpose((1, 2, 0)), lam).transpose((2, 0, 1))

        return (den_zx + den_zy + den_xy) / 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    path = pathlib.Path(args.measure)
    measure = np.load(path)
    path = path.parent
    modu_mat = np.load(path / "illumination.npy")

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

    if (path / "solution.npy").exists():
        logging.info("Results exist, skipping")
        exit(0)

    target_res = config["target_res"]
    na = config["NA"]
    model = config["model"]
    denoise = config["denoise"]

    sigma = config["sigma"]
    rho = config["rho"]
    niter = config["niter"]
    lamZ = config["lamZ"]
    lamXY = config["lamXY"]
    rtol = config["rtol"]

    psf = np.load(data_path / "psf.npy")
    spim = np.load(data_path / "spim.npy")

    if na > 0:
        psf = optics.gaussian_psf(
            numerical_aperture=na, npix_axial=127, npix_lateral=127, wavelength=600e-9
        ).squeeze()

    if spim.shape[0] < psf.shape[0]:
        crop = (psf.shape[0] - spim.shape[0]) // 2
        psf = psf[crop:-crop]
    spim_z = spim[:, spim.shape[-2] // 2, spim.shape[-1] // 2]
    psf *= spim_z[:, None, None]

    psf_z = psf[:, psf.shape[-2] // 2, psf.shape[-2] // 2]

    modu_mat = np.repeat(modu_mat, target_res // modu_mat.shape[-1], -1)

    measure = measure / measure.max()

    if model == "1D":
        modu_mat = ndimage.convolve1d(modu_mat, psf_z, axis=-1, mode="constant", cval=0)

    modu_mat = snp.asarray(modu_mat)
    measure = snp.asarray(measure)

    target_shape = list(measure.shape)
    target_shape[0] = target_res
    target_shape = tuple(target_shape)

    if model == "3D":
        convolve_op = linop.Convolve(snp.asarray(psf), target_shape, mode="same")
        sampling = ScanImaging(modu_mat, target_shape)
        sampling = linop.ComposedLinearOperator(sampling, convolve_op)

        scaling = (sampling.T @ measure).max()

        sampling = ScanImaging(modu_mat / scaling, target_shape)
        sampling = linop.ComposedLinearOperator(sampling, convolve_op)

    elif model == "1D" or model == "None":
        sampling = ScanImaging(modu_mat, target_shape)
        scaling = (sampling.T @ measure).max()
        sampling = ScanImaging(modu_mat / scaling, target_shape)

    else:
        raise ValueError(f"Invalid model: {model}")

    f = loss.SquaredL2Loss(y=measure, A=sampling)

    if denoise == "BM4D":
        from jax.lib import xla_bridge

        assert xla_bridge.get_backend().platform == "cpu"

        g = [sigma * functional.BM4D()]
        C = [linop.Identity(target_shape)]

    elif denoise == "CNN":
        g = [sigma * VDnCNN()]
        C = [linop.Identity(target_shape)]

    elif denoise == "TV":
        dXYZ = linop.FiniteDifference(input_shape=target_shape, append=0)
        g1 = functional.L21Norm()

        g = [lamZ * g1]
        C = [dXYZ]

    elif denoise == "L1":
        g1 = functional.L1Norm()

        g = [lamZ * g1]
        C = [linop.Identity(target_shape)]

    elif denoise == "TV2":
        dZ = linop.FiniteDifference(input_shape=target_shape, axes=0)
        dXY = linop.FiniteDifference(input_shape=target_shape, axes=(1, 2), append=0)

        g1 = functional.L1Norm()
        g2 = functional.L21Norm()

        g = [lamZ * g1, lamXY * g2]
        C = [dZ, dXY]

    else:
        raise NotImplementedError()

    x0 = sampling.T @ measure
    x0 = snp.mean(x0, 0)
    x0 = snp.repeat(x0[None, ...], target_shape[0], axis=0)

    rho = [rho] * len(g)

    variations = []

    def early_stop(opti):
        var = snp.abs(opti.x - opti.old_x).max() / opti.old_x.max()
        variations.append(var)

        if var < rtol:
            raise Exception("Early convergence")

        opti.old_x = opti.x

    if rtol <= 0:
        early_stop = None

    solver = admm.ADMM(
        f=f,
        g_list=g,
        C_list=C,
        rho_list=rho,
        x0=x0,
        maxiter=niter,
        subproblem_solver=admm.LinearSubproblemSolver(
            cg_kwargs={"tol": 1e-3, "maxiter": 100}
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
            ["%-*s" % (fl, fn) for fl, fn in zip(itst.fieldlength, itst.fieldname)]
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
        vari = "\n".join([f"{it} \t {var}" for (it, var) in enumerate(variations)])
        logging.info("Variations")
        logging.info(vari)

    sol = np.clip(sol, 0, None)
    cost = float(f(sol))

    out_file = path / "solution.npy"
    np.save(out_file, sol)

    out_file = path / "modulation.npy"
    np.save(out_file, modu_mat)

    stats = {"cost": cost, "time": solver.timer.elapsed(), "converged": converged}

    with (path / "stats.json").open("w") as fp:
        json.dump(stats, fp)

    try:
        sample = config["sample"]
    except KeyError:
        logging.warning("No ground truth, skipping psnr")
        exit(0)

    sample = np.load(data_path / f"{sample}.npy")
    ref_measure = np.load(path / "reference.npy")
    resolution = sample.shape[0]

    ref_psnr = cbir.psnr(sample, ref_measure, "mse")

    if target_res < resolution:
        downsize = resolution // target_res
        sample = skimage.transform.downscale_local_mean(sample, (downsize, 1, 1))

    psnr = cbir.psnr(sample, sol, norm="mse")

    if denoise == "TV2":
        regZ = float(g1(C[0] @ sol))
        regXY = float(g2(C[1] @ sol))
    else:
        regZ = 0
        regXY = 0

    factor = resolution // measure.shape[0]
    downsampled = ref_measure[::factor]

    x = np.arange(downsampled.shape[0]) * factor

    interp_f = interpolate.interp1d(
        x,
        downsampled,
        kind="cubic",
        axis=0,
        copy=False,
        fill_value=0,
        bounds_error=False,
    )

    downsampled = interp_f(np.arange(target_res) * (resolution // target_res))

    out_file = path / "undersampled.npy"
    np.save(out_file, downsampled)

    down_psnr = cbir.psnr(sample, downsampled, "mse")

    results = {
        "psnr": psnr,
        "SPIM": ref_psnr,
        "undersampled": down_psnr,
        "regZ": regZ,
        "regXY": regXY,
        "cost": cost,
        "time": solver.timer.elapsed(),
    }

    with (path / "results.json").open("w") as fp:
        json.dump(results, fp)
