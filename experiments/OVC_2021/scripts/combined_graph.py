# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
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

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("svg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["pdf.fonttype"] = 42

path = os.environ["OVC_PATH"]
gpath = os.path.join(path, "graph")

with open(os.path.join(gpath, "results.json"), "r") as fp:
    results = json.load(fp)

nas = (30, 50)
dnas = np.arange(10, 101, 5) / 100

radon_snr = results["radon"]
fss_snr = results["fss"]
fps_snr = results["fps"]
dc_snr = results["dc"]
fdc_snr = results["fdc"]

fig, plots = plt.subplots(2, len(nas), figsize=(5.5, 4.5), sharey=True, sharex=True)
lgs = None

for plot, na in zip(plots[0], nas):

    lgs = []
    na_idx = int((na - dnas[0] * 100) / 5)

    lgs.append(
        plot.hlines(radon_snr, dnas[0], dnas[-1], "C3", linestyles=":", label="X-Ray")
    )
    lgs.append(
        plot.hlines(
            fps_snr[str(na)],
            dnas[0],
            dnas[-1],
            "C2",
            linestyles="--",
            label="FPS-OPT, no deconv",
        )
    )

    lgs.append(plot.plot(dnas, dc_snr[str(na)], label="FPS-OPT, BW deconv")[0])
    lgs.append(
        plot.plot(
            dnas,
            fdc_snr[str(na)],
            "C4",
            label="FPS-OPT, GBM deconv",
            linestyle=":",
            marker=".",
            markersize=6,
        )[0]
    )

    lgs.append(
        plot.hlines(
            fss_snr[str(na)], dnas[0], dnas[-1], "C1", linestyles="-.", label="FSS-OPT"
        )
    )

    lgs.append(
        plot.plot(
            na / 100,
            dc_snr[str(na)][na_idx],
            "*",
            color="0.1",
            markersize=9,
            label="Imaging NA",
        )[0]
    )

    lgs.insert(0, lgs.pop())

    plot.set_title("Imaging NA={}".format(na / 100))
    plot.grid()
    plot.set_xlim((dnas[0], dnas[-1]))


with open(os.path.join(gpath, "noise.json"), "r") as fp:
    results = json.load(fp)

radon_snr = results["radon"]
fss_snr = results["fss"]
fps_snr = results["fps"]
dc_snr = results["dc"]
fdc_snr = results["fdc"]

for (plot, na) in zip(plots[1], nas):

    na_idx = int((na - dnas[0] * 100) / 5)

    plot.hlines(radon_snr, dnas[0], dnas[-1], "C3", linestyles=":", label="X-Ray")
    plot.hlines(
        fss_snr[str(na)], dnas[0], dnas[-1], "C1", linestyles="-.", label="FSS-OPT"
    )

    plot.plot(dnas, dc_snr[str(na)], label="FPS-OPT, BW")
    plot.plot(
        dnas,
        fdc_snr[str(na)],
        "C4",
        label="FPS-OPT, GBM",
        linestyle=":",
        marker=".",
        markersize=6,
    )

    plot.hlines(
        fps_snr[str(na)],
        dnas[0],
        dnas[-1],
        "C2",
        linestyles="--",
        label="FPS-OPT, no deconv",
    )

    plot.plot(
        na / 100,
        dc_snr[str(na)][na_idx],
        "D",
        color="0.1",
        markersize=6,
        label="Imaging NA",
    )

    plot.set_xlabel("Filtering NA")
    plot.grid()
    plot.set_xlim((dnas[0], dnas[-1]))


plots[1][0].legend(
    handles=lgs,
    loc="upper center",
    bbox_to_anchor=(0, -0.46, 1.89, 0.2),
    framealpha=1,
    ncol=3,
)

plots[0][0].set_ylabel("PSNR [dB], clean")
plots[1][0].set_ylabel("PSNR [dB], noisy")

plt.subplots_adjust(
    wspace=0.04, hspace=0.07, left=0.09, right=0.975, top=0.95, bottom=0.21
)
plt.savefig(os.path.join(gpath, "combined.pdf"))
