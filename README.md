CBI Toolbox for python
======================

Collection of algorithms used for computational bioimaging and microscopy.


Install instructions
-------------------

### Using conda (recommended):

- Create a new environment unsing the envoronment.yml file: 
  `conda env create -f environment.yml -n <environment name>`
- Run pip install on the root folder: `pip install .`


### Using pip:

- The PyPI installation of pybind11 lacks the CMake files. Add them manually,
  or compile it from source.
- Run pip install in the root folder.


CUDA support
------------

If compiling with CUDA support, make sure your CUDAToolkit_ROOT points to the
correct toolkit if you have multiple versions, or if CMake does not find it
automatically.

To debug potential installation errors, use `pip install . -v` to get verbose
build logs.

--François Marelli <francois.marelli@idiap.ch>
