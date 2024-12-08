# Size-aware Compression of 3D Gaussians with Hierarchical Mixed Precision Quantization

## 1. Cloning the Repository

```shell
git clone https://github.com/ShuzhaoXie/sizegs.git
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

CUDA 11.6/11.8: 

```shell
conda env create --file environment.yml
conda activate sizegs
pip install plyfile tqdm einops scipy open3d trimesh Ninja seaborn loguru pandas
```

CUDA 12.1/12.4:

```shell
conda create -n sizegs python=3.10
conda activate sizegs
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter
pip install tqdm plyfile einops wandb lpips laspy
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/octree
pip install submodules/quant
pip install colorama jaxtyping opencv-python tensorboard loguru pulp Ninja open3d
```

### 2.3 Preparing dataset and pre-trained 3D Gaussians

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

Then, create a ```outputs/``` folder inside the project path by 

```
mkdir outputs
```

The output structure will be organised as follows:

```
outputs/
├── dataset_name
│   ├── scene1/
│   │   ├── baseline/
│   │   │   ├── cameras.json
|   |   |   ├── cfg_args
|   |   |   ├── input.ply
|   |   |   └── point_cloud
|   |   |       ├── iteration_30000
|   |   |       │   └── point_cloud.ply
|   |   |       └── iteration_7000
|   |   |          └── point_cloud.ply
│   │   ├── sizegs/
│   │       └── the output of the sizegs
│   ├── scene2/
│   │   ├── baseline/
│   │   │   ├── cameras.json
|   |   |   ├── cfg_args
|   |   |   ├── input.ply
|   |   |   └── point_cloud
|   |   |       ├── iteration_30000
|   |   |       │   └── point_cloud.ply
|   |   |       └── iteration_7000
|   |   |          └── point_cloud.ply
│   │   ├── sizegs/
│   │       └── the output of the sizegs
...
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
This work is supported in part by National Key Research and Development Project of China (Grant No. 2023YFF0905502), Shenzhen Science and Technology Program (Grant No. JCYJ20220818101014030). We thank the valuable advices from anonymous ICLR reviewers.