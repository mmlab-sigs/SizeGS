#!/bin/bash

name=$1
iter=$2
length=$3
voxel=$4
log=$5
block=$6
size=$7
time=$(date "+%Y-%m-%d_%H:%M:%S")
logdir=sizegs_gpcc

CUDA_VISIBLE_DEVICES=1 python meson.py \
    --eval -s ~/storage/nerf_data/${log}/${name} \
    --lod 0 \
    --voxel_size ${voxel} \
    --target_size $size \
    --update_init_factor 4 \
    --appearance_dim 0 \
    --ratio 1 \
    --iterations $iter \
    --position_lr_max_steps $iter\
    --offset_lr_max_steps $iter\
    --mlp_opacity_lr_max_steps $iter\
    --mlp_cov_lr_max_steps $iter\
    --mlp_color_lr_max_steps $iter\
    --mlp_featurebank_lr_max_steps $iter\
    --appearance_lr_max_steps $iter\
    --unit_length ${length}\
    --load_iter 30000 \
    --port 8989 \
    --n_block $block \
    --mesongs \
    --raht \
    --use_pcc \
    --load_path outputs/${log}/${name}/baseline/scaffold \
    -m outputs/${log}/${name}/${logdir}/${time}
