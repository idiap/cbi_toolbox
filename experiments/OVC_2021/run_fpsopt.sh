#!/usr/bin/bash
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

scripts=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/scripts/

$OVC_PYTHON $scripts/gen_fpsopt.py

for i in {1..57}
do
    $OVC_PYTHON $scripts/gen_deconv.py $i
done

for i in {1..57}
do
    $OVC_PYTHON $scripts/recons_deconv.py $i
done

for i in {1..57}
do
    $OVC_PYTHON $scripts/fps_noise.py $i
done
