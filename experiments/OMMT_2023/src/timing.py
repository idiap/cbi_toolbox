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
    parser.add_argument("config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    config_file = pathlib.Path(args.config)
    path = config_file.parent

    if args.seed is None:
        seed = 0
    else:
        seed = args.seed
        path = path / f"{seed:02d}"
        path.mkdir(exist_ok=True)

    logging.basicConfig(filename=str(path / "log.txt"), level=logging.INFO)

    with config_file.open("r") as fp:
        config = json.load(fp)

    logging.info(f"Using config:\n{config}")
    print(f"Using config:\n{config}")

    if (path / "stats.json").exists():
        logging.info("Results exist, skipping")
        exit(0)

    width = config["width"]
    compression = config["compression"]
    denoise = config["denoise"]
    niter = config["niter"]
    warmup = config["warmup"]

    depth = width
    n_mod = depth // compression

    sigma = 0.1
    rho = 1
    lamZ = 0.1
    lamXY = 0.1

    rng = np.random.default_rng()

    modu_mat = rng.uniform(0, 1, (n_mod, depth))
    modu_mat = snp.asarray(modu_mat)

    measure = rng.uniform(0, 1, (n_mod, width, width))
    measure = snp.asarray(measure)

    target_shape = (depth, width, width)

    sampling = ScanImaging(modu_mat, target_shape)

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
        dZ = linop.FiniteDifference(input_shape=target_shape, axes=0)
        dXY = linop.FiniteDifference(input_shape=target_shape, axes=(1, 2), append=0)

        g1 = functional.L1Norm()
        g2 = functional.L1Norm()

        g = [lamZ * g1, lamXY * g2]
        C = [dZ, dXY]

    elif denoise == "L1":
        dXY = linop.FiniteDifference(input_shape=target_shape, axes=(1, 2), append=0)

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

    solver = admm.ADMM(
        f=f,
        g_list=g,
        C_list=C,
        rho_list=rho,
        x0=x0,
        maxiter=niter,
        subproblem_solver=admm.LinearSubproblemSolver(
            cg_kwargs={"tol": 1e-15, "maxiter": 20}
        ),
        itstat_options={"display": True},
    )

    for _ in range(warmup):
        solver.step()

    solver.x = x0
    sol = solver.solve()

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

    stats = {"time": solver.timer.elapsed() / niter}

    with (path / "stats.json").open("w") as fp:
        json.dump(stats, fp)
