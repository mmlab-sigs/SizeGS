#!/bin/bash

name=$1
iter=$2
prune_iter=$3
octree_iter=$4
voxel=$5
log=$6
block=$7
size=$8
device=$9
prune_per=0.6
time=$(date "+%Y-%m-%d_%H:%M:%S")
logdir=aba_iterations
csv_path=exp_data/$log/$logdir.csv

mkdir exp_data/${log}

SECONDS=0
CUDA_VISIBLE_DEVICES=$device python meson.py \
    --eval -s /mnt/storage/users/szxie_data/nerf_data/${log}/${name} \
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
    --load_iter 30000 \
    --port 8989 \
    --n_block $block \
    --mesongs \
    --use_pcc \
    --enable_drop \
    --percent $prune_per \
    --prune_iterations $prune_iter \
    --octree_iterations $octree_iter \
    --bit_upper_bound 16 \
    --bit_lower_bound 1 \
    --scene_name $name \
    --dis_type 3 \
    --csv_path $csv_path \
    --load_path outputs/${log}/${name}/baseline/scaffold \
    -m outputs/${log}/${name}/${logdir}/${time}
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."

    # --raht \
    # --save_raht \
    # --use_coarse_bit_only \ß