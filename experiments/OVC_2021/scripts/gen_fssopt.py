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
import argparse
from cbi_toolbox.utils import ome
from cbi_toolbox.simu import imaging

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)


path = os.environ["OVC_PATH"]

ipath = os.path.join(path, "imaging")


phantom = np.load(os.path.join(path, "arrays", "phantom.npy"))
spim = np.load(os.path.join(path, "arrays", "spim.npy"))

theta = np.linspace(0, 180, 360, endpoint=False)


args = parser.parse_args()
id = args.id - 1

na = (30, 50, 80)[id]

psf, _ = ome.load_ome_tiff(os.path.join(path, "psf", "BW_{:03d}.tif".format(na)))

fssopt = imaging.fss_opt(phantom, psf, spim, theta=theta)
np.save(os.path.join(ipath, "fssopt_{:03d}.npy".format(na)), fssopt)
print("Saved FSS-OPT")
