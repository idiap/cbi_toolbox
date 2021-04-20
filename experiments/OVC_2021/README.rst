OVC_2021
--------

This directory contains scripts to reproduce results published in 

    Marelli, F. and Liebling, M. *"Optics versus computation: influence of
    illumination and reconstruction model accuracy in Focal-plane-scanning optical
    projection tomography."* 2021 IEEE 18th International Symposium on Biomedical
    Imaging (ISBI). IEEE, 2021.

To run the experiments, first set the following environment variables:

* ``OVC_PATH``: full path to the directory in which you want the experiments to run (must be writable, 90G free space required)
* ``OVC_PYTHON``: full path to the python executable that will be used for the experiments

The script ``setup.sh`` creates the directory layout and generates the simulation datasets (phantom, PSF, light sheet illumination).

The script ``run_fpsopt.sh`` simulates FPS-OPT imaging, deconvolution and reconstruction with multiple PSF models (incorrect NA, approximate model).

The script ``run_fssopt.sh`` simulates FSS-OPT and X-ray imaging, and reconstruction it can be run in parallel to the previous one.

The script ``run_results.sh`` gathers the simulations and computes PSNR and generates the graphs. It requires ``matplotlib_scalebar`` to be installed.

Final results are stored in the folder ``$OVC_PATH/graph``. The `json` files contain the computed PSNR.

The simulations benefit greatly from running on multithreaded machines.
