"""
Setup script for ParticleRigidityCalculationTools.

This module handles the installation and distribution of the
ParticleRigidityCalculationTools package, a specialized library for
space physics calculations related to charged particle motion in
magnetic fields.

The package provides tools for converting between particle kinetic energy
and magnetic rigidity, which are essential parameters for studies of
cosmic rays, solar energetic particles, and magnetospheric physics.
"""

from setuptools import find_packages, setup
import os

# get requirements for installation
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='ParticleRigidityCalculationTools',
    py_modules=["ParticleRigidityCalculationTools"],
    version='1.5.17',
    description='Python library containing tools for dealing with conversions between particle energy and rigidity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Space Environment and Protection Group, University of Surrey',
    url='https://github.com/ssc-maire/ParticleRigidityCalculationTools',
    keywords='space physics earth geomagnetic rigidity magnetocosmics cosmic-rays solar-particles',
    license='CC BY-NC-SA 4.0',
    install_requires=['numpy>=1.23.1',
                      'numba>=0.57.0',
                      'setuptools>=45.2.0'],
    #install_requires,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: Creative Commons License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
