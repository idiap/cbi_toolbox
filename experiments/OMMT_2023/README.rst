OMMT_2023
---------

This directory contains scripts to reproduce results published in 

    François Marelli and Michael Liebling, *"Efficient compressed sensing 
    reconstruction for 3D fluorescence microscopy using OptoMechanical Modulation 
    Tomography (OMMT) with a 1+2D regularization"*, Opt. Express 31, 31718-31733 
    (2023) https://doi.org/10.1364/OE.493611


You will need to setup a python 3.8 environment with the requirements listed in
``requirements.txt``. Check the JAX documentation to install with CUDA support.

``pip install -r requirements.txt``

To run the experiments, first set the following environment variables:

* ``OMMT_PATH``: full path to the directory in which you want the experiments to run
* ``OMMT_PYTHON``: full path to the python executable that will be used for the experiments
* ``OMMT_DATA``: full path to the dataset (download from https://www.idiap.ch/en/dataset/ommt-fibre)
  
The script ``run_simus.sh`` runs all the simulations and generates results per
type of experiment in files named ``results.npz`` and ``lsurface.npz``. They
contain the PSNR of the solution, as well as for the reference and undersampled
SPIM, along with the time to converge and the values of the cost and
regularization (if applicable). 

The script ``run_exp.sh`` runs measurements on experimental data and
generates results in a similar fashion. 

The script ``run_timing.sh`` runs scaleability measurements on large volumes

We recommend adapting the scripts to run experiments on a computation grid for
speed, or running subsets of the experiments (comment out unneeded experiments
in the ``configure_*.py`` files).

You can read the npz files using numpy.