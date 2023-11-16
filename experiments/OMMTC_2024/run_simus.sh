#!/usr/bin/bash
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

OMMTC_SRC=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/src/

$OMMTC_PYTHON $OMMTC_SRC/gen_samples.py $OMMTC_PATH
$OMMTC_PYTHON $OMMTC_SRC/configure.py $OMMTC_PATH

find $OMMTC_PATH -name "config.json" -exec $OMMTC_PYTHON $OMMTC_SRC/simu_imaging.py {} \;
find $OMMTC_PATH -name "measure.npz" -exec $OMMTC_PYTHON $OMMTC_SRC/reconstruct.py {} \;

$OMMTC_PYTHON $OMMTC_SRC/gather_results.py $OMMTC_PATH/hollow
$OMMTC_PYTHON $OMMTC_SRC/gather_results.py $OMMTC_PATH/c8_hollow