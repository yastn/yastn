# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
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

setup(name='merge_to_matrix_cpp_1d',
    ext_modules=[cpp_extension.CppExtension(
        'merge_to_matrix_cpp_1d',
        ['torch_mtm_1d.cpp'],
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
