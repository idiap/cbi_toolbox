***********
CBI Toolbox
***********

CBI toolbox is a collection of algorithms used for computational bioimaging and microscopy.


Install instructions
====================

Requirements:

- Python>=3.7
- OpenMP>=4.5 (i.e. gcc>=6)
- MPI (optional, for distributed module)
- CudaToolkit>=8 (optional, for GPU support)


Using conda
-----------

- Clone the project with its submodules: ``git clone --recursive <url>``
- Create a new environment unsing the envoronment.yml file: 
  ``conda env create -f environment.yml -n <environment name>``
- Run pip install on the root folder: ``pip install .``

If you already have an MPI implementation installed on your system, it is possible
that conda installs a different one. If you want compatibility with your system MPI,
uninstall the conda ``mpi4py`` and ``mpi`` packages, then install ``mpi4py`` using pip. It
will automatically use your system's MPI version for compilation.


Using pip
---------

Pip >=10 is highly recommended to ensure the install works.

- Clone the project with its submodules: ``git clone --recursive <url>``
- Run pip install in the root folder: ``pip install .``


CUDA support
============

Installing with CUDA support requires the machine to have nvcc installed.
If compiling with CUDA support, make sure your CUDAToolkit_ROOT points to the
correct toolkit if you have multiple versions, or if CMake does not find it
automatically.

To debug potential installation errors, use ``pip install . -v`` to get verbose
build logs.

After install, run the following::


	import cbi_toolbox.splineradon as spl
	spl.is_cuda_available(True)


If the output is other than ``CUDA support is not installed.``, the CUDA acceleration
was installed successfully.

--François Marelli <francois.marelli@idiap.ch>
