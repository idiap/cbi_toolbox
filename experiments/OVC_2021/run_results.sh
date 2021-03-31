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

# Compute PSNR on noise-free simulations
$OVC_PYTHON ./scripts/gen_results.py

# Compute PSNR on noise-free simulations
$OVC_PYTHON ./scripts/noise_results.py

# Generate figures
$OVC_PYTHON ./scripts/combined_graph.py
$OVC_PYTHON ./scripts/noise_images.py
