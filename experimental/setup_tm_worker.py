# to build tm_worker.so in Linux:
#
# python3 -m setup_tm_worker.py build_ext --inplace
# mv tm_worker.*.so tm_worker.so
#
# A simpler option is to run   make  in this directory

from setuptools import Extension, setup

setup(
    ext_modules=[
      Extension(
        name = 'tm_worker',
        sources = ['tm_worker.c'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-lgomp']
        ),
    ]
)
