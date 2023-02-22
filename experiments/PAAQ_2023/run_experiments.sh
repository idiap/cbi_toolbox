#!/usr/bin/bash
# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
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

PAAQ_SRC=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/src/

# Convert and crop data
$PAAQ_PYTHON $PAAQ_SRC/experimental/crop_data.py $PAAQ_DATA/images/c_ventral n_frames --output_path $PAAQ_PATH/exp

# Configure experiments
find $PAAQ_PATH/exp -mindepth 1 -maxdepth 1 -type d -exec $PAAQ_PYTHON $PAAQ_SRC/experimental/configure_exp.py {} \;

# Run algorighms
find $PAAQ_PATH/exp -name "data.npz" -exec $PAAQ_PYTHON $PAAQ_SRC/algorithms/mutual_sort.py {} \;
find $PAAQ_PATH/exp -name "data.npz" -exec $PAAQ_PYTHON $PAAQ_SRC/algorithms/mcvhf_sort.py {} \;