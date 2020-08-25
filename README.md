CBI Toolbox for python
======================

Collection of algorithms used for computational bioimaging and microscopy.


Install instructions
-------------------

Install CMake version >= 3.18 (possible with anaconda).
The PyPI installation of pybind11 lacks the cmake files. Add them manually, install with anaconda (or another package manager), or compile it from source.
Run pip install in the root folder.
If compiling with CUDA support, make sure your CUDAToolkit_ROOT points to the correct toolkit (or trust CMake to find it).

--François Marelli <francois.marelli@idiap.ch>
