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

# Configure simulations
$PAAQ_PYTHON $PAAQ_SRC/simulations/configure_simu.py --outdir $PAAQ_PATH

run_pipeline () {
    # add --run-id and run multiple times for multiple random seeds
    $PAAQ_PYTHON $PAAQ_SRC/simulations/run_simu.py $1

    EXP_DIR=$(dirname $1)

    find $EXP_DIR -name "heart_simulation.npz" -exec $PAAQ_PYTHON $PAAQ_SRC/algorithms/mcvhf_sort.py {} \;

}

# Run
find $PAAQ_PATH/simu -name "config.json" -print0 | while IFS= read -r -d '' file
do run_pipeline "$file"
done

# Configure mutual info
$PAAQ_PYTHON $PAAQ_SRC/simulations/configure_mutual.py --outdir $PAAQ_PATH

run_pipeline () {
    # add --run-id and run multiple times for multiple random seeds
    $PAAQ_PYTHON $PAAQ_SRC/simulations/run_mutual.py $1

    EXP_DIR=$(dirname $1)

    find $EXP_DIR -name "heart_mutual.npz" -exec $PAAQ_PYTHON $PAAQ_SRC/algorithms/mutual_sort.py {} \;

}

# Run
find $PAAQ_PATH/mutual -name "config.json" -print0 | while IFS= read -r -d '' file
do run_pipeline "$file"
done

$PAAQ_PYTHON $PAAQ_SRC/simulations/gather_results.py ${PAAQ_PATH}/simu
$PAAQ_PYTHON $PAAQ_SRC/simulations/gather_results.py ${PAAQ_PATH}/mutual
