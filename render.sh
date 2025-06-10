#!/bin/bash

# 从外部传递的参数获取 name, percent, iter
name=$1
log=$2
iter=$3
loadpath=$4
# 记录当前时间
logdir=aba_iterations_6000

CUDA_VISIBLE_DEVICES=3 python render.py \
    --eval -s ~/nerf_data/${log}/${name} \
    --lod 0 \
    --iteration $iter\
    --update_init_factor 4 \
    --appearance_dim 0 \
    --ratio 1 \
    --mesongs \
    --raht \
    --debug \
    --skip_test \
    --name $name \
    --load_path $loadpath \
    -m $loadpath

# --input_path outputs/${log}/${name}/${logdir}/render/train

