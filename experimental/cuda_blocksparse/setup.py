# From https://github.com/pytorch/extension-cpp
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "cublocksparse"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = 1 #os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    this_dir = os.path.dirname(os.path.curdir)
    abs_dir = os.path.abspath(os.path.dirname(__file__))

    extra_link_args = [
      f"-L{os.path.join(abs_dir, '../libcutensor_nda_blocksparse/libcutensor/lib/12')}", 
      "-lcutensor",
      # "-Wl,-rpath," + os.path.join(abs_dir, "../libcutensor_nda_blocksparse/libcutensor/lib/12"),
      # "-Wl,-rpath," + os.path.join(abs_dir, "../libcutensor_nda_blocksparse/libcutensor/lib/12/"),
      # "-Wl,-rpath," + os.path.join(abs_dir, "../libcutensor_nda_blocksparse/libcutensor/lib/12/libcutensor.so"),
    ]
    # additional include dirs, IMPORTANT to be searched first
    include_dirs = [
        f"{os.path.join(abs_dir, '../libcutensor_nda_blocksparse/libcutensor/include')}",
    ]
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-std=c++17", 
            # f"-L{os.path.join(abs_dir, '../libcutensor_nda_blocksparse/libcutensor/lib/12')}", 
            # "-lcutensor",
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            include_dirs= include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="PyTorch CUDA block-sparse extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="NA",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)