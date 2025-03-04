import os
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

_src_path = osp.dirname(osp.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    '-use_fast_math'
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

setup(
    name='segquant',
    version='1.0',
    author='ShuzhaoXie',
    author_email='xiesz99@gmail.com',
    description='segquant',
    long_description='segquant',
    ext_modules=[
        CUDAExtension(
            name='segquant',
            sources=[osp.join(_src_path, 'src', f) for f in [
                'seg_quant.cu',
                'bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)