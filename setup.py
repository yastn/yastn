"""Setup script for yastn."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('YASTN - Yet Another Symmetric Tensor Network')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

def read_version():
    version_file_path = os.path.join(os.path.dirname(__file__), 'yastn', '__init__.py')
    with open(version_file_path) as fp:
        for l in fp.readlines():
            if "__version__" in l:
                return l[l.find('=')+1:].strip()

setup(
    name='yastn',
    version=read_version(),
    author='The YASTN Authors',
    author_email='marek.rams@uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.9'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude='tests')
)
