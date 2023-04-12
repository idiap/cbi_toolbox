PAAQ_2023
---------

This directory contains scripts to reproduce results published in 

    Marelli, F. and Ernst, A. and Mercader, N. and Liebling, M. *"PAAQ: Paired
    Alternating Acquisitions for Virtual High Framerate Multichannel Cardiac
    Fluorescence Microscopy"* 

To run the experiments, first set the following environment variables:

* ``PAAQ_PATH``: full path to the directory in which you want the experiments to run
* ``PAAQ_PYTHON``: full path to the python executable that will be used for the experiments
* ``PAAQ_DATA``: full path to the dataset (download from https://www.idiap.ch/en/dataset/PAAQ-Heart)
* ``PAAQ_DATA_B``: full path to the high-speed dataset (if available)
  
The script ``run_simus.sh`` runs all the simulations and generates results per type of experiment in files named ``results.npz``. They contain the phase and frequency errors, as well as metrics from the minimization algorithm in the different simulated scenarios. The outputs will be located under ``$PAAQ_PATH/simu`` and ``$PAAQ_PATH/mutual`` for sorting-based and mutual info-based results, each of the subdirectories representing one varying parameter in the experiment and containing a result file.

The script ``run_experiments.sh`` runs all methods on experimental data and generates results in files named ``*_solution.npz``. They contain the estimated phases (with * being the method used for estimation). It requires the dataset to be downloaded first to work. The outputs will be located under ``$PAAQ_PATH/exp``, each subfolder corresponding to a different crop.

The script ``run_validation.sh`` runs validation experiments on high-speed data and generates results in files named ``results.npz``. They contain the phase and frequency errors, as well as metrics from the minimization algorithm for the different parameters used. The outputs will be located under ``$PAAQ_PATH/ref``, each of the subdirectories representing one varying parameter in the experiment and containing a result file.

You can read the npz files using numpy.