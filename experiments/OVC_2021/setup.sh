#!/bin/bash
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

# Create all experiment directories
mkdir ${OVC_PATH}
mkdir ${OVC_PATH}/arrays
mkdir ${OVC_PATH}/deconv
mkdir ${OVC_PATH}/noise
mkdir ${OVC_PATH}/reconstruct
mkdir ${OVC_PATH}/imaging
mkdir ${OVC_PATH}/psf

scripts=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )/scripts/

# Generate 3D phantom for simulations and SPIM illumination
$OVC_PYTHON $scripts/gen_phantom.py
$OVC_PYTHON $scripts/gen_spim.py

# Generate PSF using Born & Wolf model
cd ${OVC_PATH}
wget http://bigwww.epfl.ch/algorithms/psfgenerator/PSFGenerator.jar

# Type of PSF generated
type='BW'

# Everything will be divided by 100
start=10
step=5

# Iterations
iter=19

path=${OVC_PATH}

if [ ! -f ${path}/PSFGenerator.jar ]; then
    echo "Please run setup.sh first."
    exit 1
fi

cd ${path}/psf
name="PSF ${type}.tif"

gen_config () {
    cat > ${path}/_config.txt <<EOF
#PSFGenerator
PSF-shortname=$type
ResLateral=635.0
ResAxial=635.0
NY=257
NX=257
NZ=257
Type=32-bits
NA=$NA
LUT=Fire
Lambda=500.0
Scale=Linear
psf-BW-NI=1.333
psf-BW-accuracy=Best
psf-RW-NI=1.333
psf-RW-accuracy=Best
EOF
}

for (( i=0; i<iter; i++ ))
do
    N=$(($start+$i*$step))
    N=$(printf "%03d\n" $N)
    NA=${N:0:1}.${N:1}

    echo "Generating for NA=${NA}"
    
    gen_config

    java -cp ${path}/PSFGenerator.jar PSFGenerator ${path}/_config.txt

    mv "$name" "${type}_${N}.tif"
    rm _config.txt
done
