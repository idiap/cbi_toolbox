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
import numpy as np
import json

from cbi_toolbox.reconstruct import psnr


nas = (30, 50, 80)
dnas = np.arange(10, 101, 5)
norm = "mse"


path = os.environ["OVC_PATH"]

rpath = os.path.join(path, "reconstruct")
gpath = os.path.join(path, "graph")

ref = np.load(os.path.join(path, "arrays", "phantom.npy"))
radon = np.load(os.path.join(rpath, "iradon.npy"))

results = {"fss": {}, "fps": {}, "dc": {}, "fdc": {}}

fss_snr = results["fss"]
fps_snr = results["fps"]
dc_snr = results["dc"]
fdc_snr = results["fdc"]

results["radon"] = psnr(ref, radon, norm)
del radon

for na in nas:

    fss = np.load(os.path.join(rpath, "fssopt_{:03d}.npy".format(na)))
    fss_snr[na] = psnr(ref, fss, norm)
    del fss

    fps = np.load(os.path.join(rpath, "fpsopt_{:03d}.npy".format(na)))
    fps_snr[na] = psnr(ref, fps, norm)
    del fps

    dc_snr[na] = []
    fdc_snr[na] = []

    for dna in dnas:
        dc = np.load(os.path.join(rpath, "{:03d}_{:03d}.npy".format(na, dna)))
        dc_snr[na].append(psnr(ref, dc, norm))

        fdc = np.load(os.path.join(rpath, "{:03d}_{:03d}f.npy".format(na, dna)))
        fdc_snr[na].append(psnr(ref, fdc, norm))


with open(os.path.join(gpath, "results.json"), "w") as fp:
    json.dump(results, fp)
