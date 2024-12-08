#!/bin/bash

# 从外部传递的参数获取 name, percent, iter
name=$1
log=$2
iter=$3
# 记录当前时间
logdir=sizegs

python render.py \
    --eval -s path/to/data/${log}/${name} \
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
    --load_path path/to/outputs/${log}/${name}/baseline/scaffold \
    -m path/to/outputs/${log}/${name}/${logdir}/render \
    --input_path path/to/outputs/${log}/${name}/${logdir}/render/train