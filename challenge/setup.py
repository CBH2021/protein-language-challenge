import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    # use environment.yml
]


setup(
    name="challenge",
    version="0.0.1",
    url="",
    author="<fill>",
    author_email="<fill>",
    description="Protein language copenhagen hackaton challenge",
    long_description=read("../README.rst"),
    packages=find_packages(exclude=()),
    entry_points={
        "console_scripts": [
            "challenge=challenge.cli:cli"
        ]
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
