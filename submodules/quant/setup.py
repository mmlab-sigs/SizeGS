import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='quant_cuda',
    version='1.0',
    author='JiahangLiu',
    author_email='Liu030526@gmail.com',
    description='quant_cuda',
    long_description='quant_cuda',
    ext_modules=[
        CUDAExtension(
            name='quant_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)