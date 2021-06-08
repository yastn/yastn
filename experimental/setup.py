from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='merge_to_matrix_cpp',
    ext_modules=[cpp_extension.CppExtension(
        'merge_to_matrix_cpp', 
        ['torch_mtm.cpp'],
        extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='mm_incommensurate_batch_cpp',
    ext_modules=[cpp_extension.CppExtension(
        'mm_incommensurate_batch_cpp', 
        ['torch_mmib.cpp'],
        extra_compile_args=['-fopenmp']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
