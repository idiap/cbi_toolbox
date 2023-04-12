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

$OMMT_PYTHON $OMMT_SRC/configure_timing.py $OMMT_PATH

# For CPU testing
find $OMMT_PATH -name "config_cpu.json" -exec $OMMT_PYTHON $OMMT_SRC/timing.py {} \;
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/t_BM --experiment
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/t_TV --experiment
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/t_L1 --experiment

# For CUDA testing
# Uncomment following and comment previous, run on a machine with CUDA capability
# find $OMMT_PATH -name "config_cuda.json" -exec $OMMT_PYTHON $OMMT_SRC/timing.py {} \;
# $OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/t_TV_cuda --experiment
# $OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/t_L1_cuda --experiment