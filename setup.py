import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pampropy",
    packages = ["pampropy"],
    version = "0.1",
    author = "Tom White",
    author_email = "thomite@gmail.com",
    description = ("Physical Activity Monitoring Processing in Python"),
    url = "https://github.com/Thomite/pampropy"
)

