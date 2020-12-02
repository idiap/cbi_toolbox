"""
CBI Toolbox contains a collection of algorithms used for computational
bioimaging and microscopy.
"""

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


import os
import platform
import subprocess
import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

__version__ = '0.1'

requires = [
    'numpy',
    'opencv-python',
    'apeer-ometiff-library',
    'scipy>=1.4.0',
    'scikit-image',
    'noise',
    'poppy',
]

extras_require = {
    'plots': ['napari', 'matplotlib'],
    'mpi': ['mpi4py'],
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--clean-first']

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if os.path.exists(self.build_temp):
            import shutil
            shutil.rmtree(self.build_temp)

        os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)


setup(
    name='cbi_toolbox',
    version=__version__,
    author='François Marelli',
    author_email='francois.marelli@idiap.ch',
    description='A python toolbox for computational bioimaging',
    long_description='',
    ext_modules=[
        CMakeExtension('cbi_toolbox.cmake_ext'),
    ],
    packages=find_packages(exclude=('tests', )),
    install_requires=requires,
    python_requires='>=3.7',
    extras_require=extras_require,
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
