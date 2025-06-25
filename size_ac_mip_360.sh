# !/bin/bash

# config 1 12 6 4 10 5 3 12 8 9
# config 2 
# config 3
# bicycle 0.0001
# bonsai 0.00008
# counter 0.00003
# garden 0.00003
# kitchen 0.0005
# room 0.00003
# stump 0.00003
# flowers 0.0001
# treehill 0.0001
# 33
# 28
# 32
# 16
# 11
# 14
# 8
# 21
# 26
# echo "Running for name: bicycle, iter: 6000"
# bash ./sizegs.sh bicycle 6000 4000 4000 0.0001 360_v2 60 33 1
# echo "Completed for name: bicycle"
# sleep 1s

echo "Running for name: garden, iter: 6000"
bash ./sizegs_torchac.sh garden 4000 4000 4000 0.001 360_v2 60 28 2 0.5 0.05
echo "Completed for name: garden"

# echo "Running for name: stump, iter: 6000"
bash ./sizegs_torchac.sh stump 4000 4000 4000 0.001 360_v2 60 32 2 0.5 0.05
echo "Completed for name: stump"


# echo "Running for name: bonsai, iter: 6000"
bash ./sizegs_torchac.sh bonsai 4000 4000 4000 0.001 360_v2 60 16 2 0.5 0.05
echo "Completed for name: bonsai"

sleep 1s

# echo "Running for name: counter, iter: 6000"
bash ./sizegs_torchac.sh counter 4000 4000 4000 0.001 360_v2 60 11 2 0.5 0.05
echo "Completed for name: counter"

# echo "Running for name: kitchen, iter: 6000"
bash ./sizegs_torchac.sh kitchen 4000 4000 4000 0.001 360_v2 60 14 2 0.5 0.05
echo "Completed for name: kitchen"

# echo "Running for name: room, iter: 6000"
bash ./sizegs_torchac.sh room 4000 4000 4000 0.001 360_v2 60 8 2 0.5 0.05
echo "Completed for name: room"


# echo "Running for name: flowers, iter: 6000"
bash ./sizegs_torchac.sh flowers 4000 4000 4000 0.001 360_v2 60 21 2 0.5 0.05
echo "Completed for name: flowers"

sleep 1s

# echo "Running for name: treehill, iter: 6000"
bash ./sizegs_torchac.sh treehill 4000 4000 4000 0.001 360_v2 60 26 2 0.5 0.05
echo "Completed for name: treehill"


#!/bin/bash
# echo "Running for name: drjohnson, iter: 8000"
bash ./sizegs.sh train 6000 4000 4000 0.001 tandt 60 7 1 0.5 0.05
# echo "Completed for name: drjohnson"

sleep 1s

# echo "Running for name: playroom, iter: 8000"
bash ./sizegs.sh truck 6000 4000 4000 0.001 tandt 60 9 1 0.5 0.05
# echo "Completed for name: playroom"

echo "Running for name: drjohnson, iter: 8000"
bash ./sizegs.sh drjohnson 4000 4000 4000 0.001 db 60 4 1 0.5 0.05
echo "Completed for name: drjohnson"

# sleep 1s

# echo "Running for name: playroom, iter: 8000"
bash ./sizegs.sh playroom 4000 4000 4000 0.001 db 60 4 1 0.5 0.05
# echo "Completed for name: playroom"
