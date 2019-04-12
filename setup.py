from setuptools import setup

requires = [
    'numpy',
    'matplotlib',
    'opencv-python',
    'python-bioformats',
    'javabridge',
    
]

setup(
    name='cbi_toolbox',
    version='0.1',
    install_requires=requires,
    packages=['cbi_toolbox'],
)
