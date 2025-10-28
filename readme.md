# SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming

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

CUDA 12.1/12.4:

```shell
conda create -n sizegs python=3.10
conda activate sizegs
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install tqdm plyfile einops wandb lpips laspy colorama jaxtyping opencv-python tensorboard loguru pulp Ninja open3d torchac
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/octree
pip install submodules/quant
pip install submodules/seg-quant
pip install numpy==1.26.4
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


## 3. Running

### Use `torchac`
* Check `size_ac_mip_360.sh`.

### Use `LZ77 (zip)`
* Check `size_mip_360.sh`.

* Check `size_tandt.sh`.

* Check `size_db.sh`.

### TODO
- [ ] lint the code.

## 4. Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{xie2024sizegs,
    title={SizeGS: Size-aware Compression of 3D Gaussian Splatting via Mixed Integer Programming},
    author={Xie, Shuzhao and Liu, Jiahang and Zhang, Weixiang and Ge, Shijia and Pan, Sicheng and Tang, Chen and Bai, Yunpeng and Zhang, Cong and Fan, Xiaoyi and Wang, Zhi},
    booktitle={ACM MM},
    year={2025}
}
```

## 5. Acknowledgments
We sincerely thank the anonymous reviewers from ICLR, CVPR, and ACM MM for their valuable feedback and suggestions. We also thank our lab mates for their help in improving the manuscript. This work is supported in part by National Key Research and Development Project of China (Grant No. 2023YFF0905502), National Natural Science Foundation of China (Grant No. 92467204 and 62472249), and Shenzhen Science and Technology Program (Grant No. JCYJ20220818101014030 and KJZD20240903102300001).
