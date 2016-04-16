import os
from setuptools import setup

setup(
    name = "pampro",
    packages = ["pampro"],
    version = "0.4",
    author = "Tom White",
    author_email = "thomite@gmail.com",
    description = ("physical activity monitor processing"),
    url = "https://github.com/Thomite/pampro",
    install_requires = ['numpy', 'scipy', 'matplotlib', 'h5py']
)
