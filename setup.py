from setuptools import setup, find_packages

setup(
    name="pyloess",
    version="v0.1.0",
    description="A simple implementation of the LOESS algorithm using numpy based on NIST",
    author="Nicolai Thomassen",
    author_email="nith@dhigroup.com",
    packages=find_packages(),
    python_requires=">=3.6",
)
