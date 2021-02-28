"""Setup script for yast."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('Basic operations for matrix product states.')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

__version__ = '0.1.0'

setup(
    name='yast',
    version=__version__,
    author='Marek M. Rams, Gabriela Wójtowicz, Piotr Czarnik, Juraj Hasik',
    author_email='marek.rams@uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.7.0'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude='tests')
)
