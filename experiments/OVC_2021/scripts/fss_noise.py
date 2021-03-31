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

import cbi_toolbox.splineradon as spl
from cbi_toolbox.simu import imaging

photons = 1e4

path = os.environ['OVC_PATH']

ipath = os.path.join(path, 'imaging')
npath = os.path.join(path, 'noise')

theta = np.linspace(0, 180, 360, endpoint=False)


for na in (30, 50, 80):
    fps_opt = np.load(os.path.join(ipath, 'fpsopt_{:03d}.npy'.format(na)))

    fps_opt = imaging.noise(fps_opt, seed=0, in_place=True, photons=photons)

    iradon = spl.iradon(fps_opt, theta=theta, circle=True)
    np.save(os.path.join(npath, 'fpsopt_{:03d}.npy'.format(na)), iradon)
    print('fpsopt saved')
    del iradon, fps_opt

    fss_opt = np.load(os.path.join(ipath, 'fssopt_{:03d}.npy'.format(na)))

    fss_opt = imaging.noise(fss_opt, seed=0, in_place=True, photons=photons)

    iradon = spl.iradon(fss_opt, theta=theta, circle=True)
    np.save(os.path.join(npath, 'fssopt_{:03d}.npy'.format(na)), iradon)
    print('fssopt saved')
    del iradon, fss_opt
