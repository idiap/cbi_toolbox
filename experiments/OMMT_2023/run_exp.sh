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


$OMMT_PYTHON $OMMT_SRC/configure_exp.py $OMMT_PATH $OMMT_DATA

run_pipeline () {
    EXP_DIR=$(dirname $1)
    EXP_DIR=$(dirname $EXP_DIR)
    EXP_DIR=$(dirname $EXP_DIR)
    EXP_DIR=$(dirname $EXP_DIR)

    $OMMT_PYTHON $OMMT_SRC/reconstruct.py $EXP_DIR/measure_exp.npy --config $1
}

find $OMMT_PATH -name "config_exp.json" -print0 | while IFS= read -r -d '' file
do run_pipeline "$file"
done

$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/exp_s1 --experiment
$OMMT_PYTHON $OMMT_SRC/gather_results_lsurface.py $OMMT_PATH/exp_s2 --experiment
