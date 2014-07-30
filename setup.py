import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pampro",
    packages = ["pampro"],
    version = "0.1",
    author = "Tom White",
    author_email = "thomite@gmail.com",
    description = ("Physical Activity Monitoring Processing"),
    url = "https://github.com/Thomite/pampro"
)

