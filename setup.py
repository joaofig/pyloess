from setuptools import setup, find_packages

setup(
    name='loess',
    version='v1',
    description='Loess interpolation for s2 regular timeseries',
    author='Nicolai Thomassen',
    author_email='nith@dhigroup.com',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[],
    extras_require={    }
)
