***********
CBI Toolbox
***********

CBI Toolbox is a collection of algorithms used for computational bioimaging and microscopy.

CBI Toolbox is suitable for large scale 3D simulations to reproduce realistic sample sizes and sampling resolution thanks to hardware acceleration (both GPU and multithreading) and optimized implementation of the algorithms.
It is flexible enough to emulate different acquisition geometries, and also contains a set of tools used for quantitative evaluation of imaging methods.
The framework can also be used for experiment design, as geometries can be virtually tested before being implemented on real acquisition devices.

.. image:: docs/cover.svg
  :alt: Demo image

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/

Written by François Marelli <francois.marelli@idiap.ch>

For documentation, see https://cbi-toolbox.readthedocs.io

Install instructions
====================

Requirements:

- Python>=3.7
- OpenMP>=2
- MPI (optional, for parallel module)
- CudaToolkit>=8 (optional, for GPU support)


Using pip
---------

Pip >=10 is highly recommended to ensure the install works.

- Run pip install: ``pip install cbi-toolbox[mpi,plots,docs,ome] @ git+https://github.com/idiap/cbi_toolbox.git`` (choose optional packages according to your needs)

**PyPI hosted version is capped to 1.1.0 due to the addition of non PyPI hosted dependencies**

Install from sources:

- Clone the project with its submodules: ``git clone --recursive <url>``
- Run pip install in the root folder: ``pip install .[mpi,plots,docs,ome]`` (choose optional packages according to your needs)


Optional dependencies
---------------------

The package provides optional dependencies that can be selected  at will during
the install (in the square brackets):

- **mpi**: allows to use the ``cbi_toolbox.parallel.mpi`` module,
  requires a functional MPI installation
- **plots**: installs tools to visualize 3D objects easily
- **docs**: installs tools used to generate the documentation
- **ome**: installs tools to read and write ome-tiff images


Using conda
-----------

- Clone the project with its submodules: ``git clone --recursive <url>``
- Create a new environment unsing the environment.yml file: 
  ``conda env create -f environment.yml -n <environment name>``
- Run pip install on the root folder: ``pip install .[mpi,plots,docs,ome]`` (choose optional packages according to your needs)

If you already have an MPI implementation installed on your system, it is possible
that conda installs a different one. If you want compatibility with your system MPI,
uninstall the conda ``mpi4py`` and ``mpi`` packages, then install ``mpi4py`` using pip. It
will automatically use your system's MPI version for compilation.


CUDA support
============

If nvcc is present on the machine, the installation will automatically attempt
to compile the software with CUDA support. If you have multiple versions of the
CUDA toolkit installed, or if CMake fails to find nvcc automatically, make sure
to set the environment variable ``CUDAToolkit_ROOT`` to point to the correct
tookit folder.

To debug potential installation errors, use ``pip install . -v`` to get verbose
build logs.

After install, run the following::


	import cbi_toolbox.splineradon as spl
	spl.is_cuda_available(True)


If the output is other than ``CUDA support is not installed.``, the CUDA acceleration
was installed successfully.


Citing CBI Toolbox
==================

If you use this package and wish to aknowledge it in an academic publication,
please cite the following paper:

    Marelli, F. and Liebling, M. *"Optics versus computation: influence of
    illumination and reconstruction model accuracy in Focal-plane-scanning optical
    projection tomography."* 2021 IEEE 18th International Symposium on Biomedical
    Imaging (ISBI). IEEE, 2021.


Acknowledgements
================

This work was supported by the Swiss National Science Foundation (SNSF), Grants:

* 206021_164022 *Platform for Reproducible Acquisition, Processing, and Sharing of Dynamic, Multi-Modal Data.* (PLATFORM_MMD)
* 200020_179217 *Computational biomicroscopy: advanced image processing methods to quantify live biological systems* (COMPBIO)