"""Setup script for yamps."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('Suplementary functions for quantum impurity transport.')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

__version__ = '0.0.1'

setup(
    name='AIM_transport',
    version=__version__,
    author='Gabriela Wójtowicz',
    author_email='g.wojtowicz@doctoral.uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.6.0'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
