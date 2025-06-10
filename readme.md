# Size-aware Compression of 3D Gaussians with Hierarchical Mixed Precision Quantization

## 1. Cloning the Repository

```shell
git clone https://github.com/mmlab-sigs/sizegs --recursive
```

## 2. Install

### 2.1 Hardware and Software Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- Ubuntu >= 18.04
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we *recommend* Visual Studio 2019 for Windows)
- CUDA >= 11.6
- C++ Compiler and CUDA SDK must be compatible

### 2.2 Setup
Our provided install method is based on Conda package and environment management:
<!-- 
CUDA 11.6/11.8: 

```shell
conda env create --file environment.yml
conda activate sizegs
pip install plyfile tqdm einops scipy open3d trimesh Ninja seaborn loguru pandas
``` -->

CUDA 12.1/12.4:

```shell
conda create -n sizegs python=3.10
conda activate sizegs
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install tqdm plyfile einops wandb lpips laspy colorama jaxtyping opencv-python tensorboard loguru pulp Ninja open3d
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/octree
pip install submodules/quant
pip install submodules/seg-quant
pip install numpy==1.26.4

cd submodules/BDD
git checkout f377a82736435bc4988e2c41e5c8029c168e9505
setup.py line 94: set 
cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
    '-DPYTHON_EXECUTABLE=' + sys.executable,
    '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc',
    '-DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.4/bin/nvcc',
    '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA'
]
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libboost-all-dev
cd ../../
pip install submodules/BDD
conda update -c conda-forge libstdcxx-ng
```

### 2.3 Preparing dataset and pre-trained ScaffoldGSes
#### 2.3.1 Dataset
First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

#### 2.3.2 Pretrained ScaffoldGSes
Then, create a ```outputs/``` folder inside the project path by 

```
mkdir outputs
```

Take T&T as an example, the ```outputs/tandt``` structure should be organized as follows. An example *lego* ScaffoldGS can be downloaded from [here](https://drive.google.com/file/d/1QcRPs6mIHaFZOCYFEzT75G8k2vkYE7vL/view?usp=drive_link). These pretrained ScaffoldGSes serve at the input of our code.

```
.
├── train
│   └── baseline
│       └── scaffold
│           ├── backup
│           ├── cameras.json
│           ├── cfg_args
│           ├── input.ply
│           ├── outputs.log
│           ├── per_view.json
│           ├── point_cloud
│           ├── results.json
│           └── test
└── truck
    └── baseline
        └── scaffold
            ├── backup
            ├── cameras.json
            ├── cfg_args
            ├── input.ply
            ├── outputs.log
            ├── per_view.json
            ├── point_cloud
            ├── results.json
            └── test
```

At last change the ```path/to/outputs/``` and ```path/to/data/``` in meson.sh with your path


## Running

Check `mesonmip_360.sh`.
Check `mesondb.sh`.
Check `mesonnerf.sh`.
Check `mesontandt.sh`.



## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{xie2024sizegs,
    title={SizeGS: Size-aware Compression of 3D Gaussians with Hierarchical Mixed Precision Quantization},
    author={Xie, Shuzhao and Liu, Jiahang and Zhang, Weixiang and Ge, Shijia and Pan, Sicheng and Tang, Chen and Bai, Yunpeng and Wang, Zhi},
    journal={arXiv},
    year={2024}
}
```

## Acknowledgments
This work is supported in part by National Key Research and Development Project of China (Grant No. 2023YFF0905502), Shenzhen Science and Technology Program (Grant No. JCYJ20220818101014030). We thank the valuable advices from anonymous ICLR and CVPR reviewers.