"""Setup script for yamps."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('Basic operations for matrix product states.')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

__version__ = '0.0.1'

setup(
    name='yamps',
    version=__version__,
    author='Gabriela Wójtowicz, Marek M. Rams',
    author_email='marek.rams@uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude='tests')
)
