OMMTC_2024
----------

This directory contains scripts to reproduce results published in 

        François Marelli and Michael Liebling, *"Optomechanical modulation
        tomography for ungated compressive cardiac light sheet microscopy"*

You will need to setup a python 3.8 environment with the requirements listed in
``requirements.txt``.

``pip install -r requirements.txt``

To run the experiments, first set the following environment variables:

* ``OMMTC_PATH``: full path to the directory in which you want the experiments to run
* ``OMMTC_PYTHON``: full path to the python executable that will be used for the experiments
  
The script ``run_simus.sh`` runs all the simulations and generates results per
type of experiment in files named ``combined.npz``. They contain the PSNR of the
solution, along with the time to converge and the values of the cost and regularization. 

We recommend adapting the scripts to run experiments on a computation grid for
speed, or running subsets of the experiments (comment out unneeded experiments
in the ``configure_*.py`` files).

You can read the npz files using numpy.
