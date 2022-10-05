from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fused_transpose_merge_1d',
    ext_modules=[cpp_extension.CppExtension(
        'fused_transpose_merge_1d', 
        ['torch_tm_1d.cpp'],
        extra_compile_args=['-fopenmp', '-march=native']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})