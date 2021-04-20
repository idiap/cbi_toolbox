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

import argparse
import os
import numpy as np
import cbi_toolbox.splineradon as spl

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int)

args = parser.parse_args()
id = args.id - 1

na = id // 19
dna = id % 19

na = (30, 50, 80)[na]
dna = 10 + 5 * dna

theta = np.linspace(0, 180, 360, endpoint=False)

path = os.environ['OVC_PATH']

dpath = os.path.join(path, 'deconv')
rpath = os.path.join(path, 'reconstruct')

for suffix in ('', 'f'):
    file = '{:03d}_{:03d}{}.npy'.format(na, dna, suffix)

    deconv = np.load(os.path.join(dpath, file))

    iradon = spl.iradon(deconv, theta=theta, circle=True)

    np.save(os.path.join(rpath, file), iradon)
    print('{} saved'.format(file))
