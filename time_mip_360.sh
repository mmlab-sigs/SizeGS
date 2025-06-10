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

#!/bin/bash
# echo "Running for name: drjohnson, iter: 8000"
bash ./sizegs.sh train 1 100 100 0.001 tandt 60 10 0
# echo "Completed for name: drjohnson"

sleep 1s

# echo "Running for name: playroom, iter: 8000"
bash ./sizegs.sh truck 1 100 100 0.001 tandt 60 12 0
# echo "Completed for name: playroom"


echo "Running for name: drjohnson, iter: 8000"
bash ./sizegs.sh drjohnson 1 100 100 0.0001 db 60 8 0
echo "Completed for name: drjohnson"

# sleep 1s

# echo "Running for name: playroom, iter: 8000"
bash ./sizegs.sh playroom 1 100 100 0.0001 db 60 8 0

# echo "Running for name: bicycle, iter: 1"
# bash ./sizegs.sh bicycle 1 100 100 0.0001 360_v2 60 24 2
# echo "Completed for name: bicycle"

# sleep 1s

# # echo "Running for name: bonsai, iter: 1"
# bash ./sizegs.sh bonsai 1 100 100 0.00008 360_v2 60 12 2
# echo "Completed for name: bonsai"

# sleep 1s

# # echo "Running for name: counter, iter: 1"
# bash ./sizegs.sh counter 1 100 100 0.00003 360_v2 60 8 2
# echo "Completed for name: counter"

# sleep 1s

# echo "Running for name: garden, iter: 1"
# bash ./sizegs.sh garden 1 100 100 0.00003 360_v2 60 20 2
# echo "Completed for name: garden"

# sleep 1s

# # echo "Running for name: kitchen, iter: 1"
# bash ./sizegs.sh kitchen 1 100 100 0.0005 360_v2 60 10 2
# echo "Completed for name: kitchen"

# sleep 1s

# # echo "Running for name: room, iter: 1"
# bash ./sizegs.sh room 1 100 100 0.00003 360_v2 60 8 2
# echo "Completed for name: room"

# sleep 1s

# # echo "Running for name: stump, iter: 1"
# bash ./sizegs.sh stump 1 100 100 0.00003 360_v2 60 24 2
# echo "Completed for name: stump"

# sleep 1s

# # echo "Running for name: flowers, iter: 1"
# bash ./sizegs.sh flowers 1 100 100 0.0001 360_v2 60 16 2
# echo "Completed for name: flowers"

# sleep 1s

# # echo "Running for name: treehill, iter: 1"
# bash ./sizegs.sh treehill 1 100 100 0.0001 360_v2 60 18 2
# echo "Completed for name: treehill"

