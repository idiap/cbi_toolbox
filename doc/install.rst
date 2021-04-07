Installation
============

Requirements:

- Python>=3.7
- OpenMP>=4.5 (i.e. gcc>=6)
- MPI (optional, for parallel module)
- CudaToolkit>=8 (optional, for GPU support)


Using conda
-----------

- Clone the project with its submodules: ``git clone --recursive <url>``
- Create a new environment unsing the envoronment.yml file: 
  ``conda env create -f environment.yml -n <environment name>``
- Run pip install on the root folder: ``pip install .[mpi,plots]``

If you already have an MPI implementation installed on your system, it is possible
that conda installs a different one. If you want compatibility with your system MPI,
uninstall the conda ``mpi4py`` and ``mpi`` packages, then install ``mpi4py`` using pip. It
will automatically use your system's MPI version for compilation.


Using pip
---------

Pip >=10 is highly recommended to ensure the install works.

- Clone the project with its submodules: ``git clone --recursive <url>``
- Run pip install in the root folder: ``pip install .[mpi,plots]``


Optional dependencies
---------------------

The package provides optional dependencies that can be selected  at will during
the install (in the square brackets):

- **mpi**: allows to use the ``cbi_toolbox.parallel.mpi`` module,
  requires a functional MPI installation
- **plots**: installs tools to visualize 3D objects easily


CUDA support
------------

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