# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
from setuptools import setup

setup(
    name="pampro",
    packages=["pampro"],
    version="0.5.1",
    author="Tom White\Ella Hutchinson",
    author_email="ella.hutchinson@mrc-epid.cam.ac.uk",
    maintainer="Ella Hutchinson",
    maintainer_email="ella.hutchinson@mrc-epid.cam.ac.uk",
    license="GNU GPL-3.0",
    description=("physical activity monitor processing"),
    url="https://github.com/MRC-Epid/pampro",
    install_requires=['numpy>=1.14.0', 'scipy>=1.1.0', 'matplotlib>=2.2.2', 'h5py>=2.9.0', 'pandas>==0.23.0', 'statsmodels>=0.9.0', 'uos_activpal>=0.2.2', 'numba>=0.45'],
    Classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows :: Windows 7",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
)
