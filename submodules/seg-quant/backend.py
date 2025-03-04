import os 
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
# "-g", "-G", 
nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
    '-use_fast_math'
]

c_flags = ['-O3', '-std=c++17']

_src_path = os.path.dirname(os.path.abspath(__file__))

os.environ['TORCH_EXTENSIONS_DIR'] = './log'

_backend = load(
    name='_segquant',
    extra_cflags=c_flags,
    extra_cuda_cflags=nvcc_flags,
    sources=[
        os.path.join(_src_path, 'src', f) for f in [
            'bindings.cpp',
            'seg_quant.cu',
        ]
    ],
)

__all__ = ['_backend']