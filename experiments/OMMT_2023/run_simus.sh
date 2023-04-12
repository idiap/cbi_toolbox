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

OMMT_SRC=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/src/

$OMMT_PYTHON $OMMT_SRC/gen_samples.py $OMMT_PATH
$OMMT_PYTHON $OMMT_SRC/configure.py $OMMT_PATH

find $OMMT_PATH -name "config.json" -exec $OMMT_PYTHON $OMMT_SRC/simu_imaging.py {} \;
find $OMMT_PATH -name "measure.npy" -exec $OMMT_PYTHON $OMMT_SRC/reconstruct.py {} \;

$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/TV_grid
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/TV_multiline
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/TV_moon

$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/nTV_grid
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/nTV_multiline

$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/BM_grid
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/BM_multiline
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/BM_moon

$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/nBM_grid
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/nBM_multiline

$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/L1_grid
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/L1_multiline
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/L1_moon

$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/nL1_grid
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/nL1_multiline

$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/TV1_multiline
$OMMT_PYTHON $OMMT_SRC/gather_results.py $OMMT_PATH/nTV1_multiline
