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


path = os.environ['OVC_PATH']

ipath = os.path.join(path, 'imaging')


phantom = np.load(os.path.join(path, 'arrays', 'phantom.npy'))

theta = np.linspace(0, 180, 360, endpoint=False)

radon = spl.radon(phantom, theta=theta, circle=True)
np.save(os.path.join(ipath, 'radon.npy'), radon)
print('Saved Radon')
