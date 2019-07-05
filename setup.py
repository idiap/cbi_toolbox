import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

__version__ = '0.1'

requires = [
    'numpy',
    'matplotlib',
    'opencv-python',
    'python-bioformats',
    'javabridge',
    'pybind11>=2.2',
    'scipy'

]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    @staticmethod
    def get_pybind_include(user):
        import pybind11
        return pybind11.get_include(user)

    @staticmethod
    def get_numpy_include():
        import numpy as np
        return np.get_include()

    @staticmethod
    def get_xtensor_include():
        return os.path.join(sys.prefix, 'include')

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPYTHON_INCLUDES=' + ';'.join((self.get_numpy_include(), self.get_pybind_include(False), self.get_pybind_include(True), self.get_xtensor_include()))]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='cbi_toolbox',
    version=__version__,
    author='François Marelli',
    author_email='francois.marelli@idiap.ch',
    description='A python toolbox for computational bioimaging',
    long_description='',
    ext_modules=[CMakeExtension('cbi_toolbox.csplineradon')],
    install_requires=requires,
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
