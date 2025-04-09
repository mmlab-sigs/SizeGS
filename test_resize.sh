#!/bin/bash

name=bicycle
iter=10
prune_iter=10
octree_iter=10
voxel=0.0001
log=360_v2
block=60
size=24
prune_per=0.6
time=$(date "+%Y-%m-%d_%H:%M:%S")
logdir=sizegs_gpcc

SECONDS=0
CUDA_VISIBLE_DEVICES=1 python research.py \
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
    --raht \
    --save_raht \
    --test_resize \
    --newsize 16 \
    --debug \
    --use_pcc \
    --percent $prune_per \
    --bit_upper_bound 16 \
    --bit_lower_bound 1 \
    --scene_name $name \
    --enable_drop \
    --prune_iterations $prune_iter \
    --octree_iterations $octree_iter \
    --load_path outputs/${log}/${name}/baseline/scaffold \
    -m outputs/${log}/${name}/${logdir}/${time}
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."

    # --enable_drop \