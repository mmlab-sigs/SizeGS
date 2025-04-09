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

# echo "Running for name: bicycle, iter: 6000"
bash ./sizegs.sh bicycle 4000 2000 2000 0.0001 360_v2 60 24
echo "Completed for name: bicycle"

sleep 1s

# echo "Running for name: bonsai, iter: 6000"
bash ./sizegs.sh bonsai 4000 2000 2000 0.00008 360_v2 60 12
echo "Completed for name: bonsai"

sleep 1s

# echo "Running for name: counter, iter: 6000"
bash ./sizegs.sh counter 4000 2000 2000 0.00003 360_v2 60 8
echo "Completed for name: counter"

sleep 1s

echo "Running for name: garden, iter: 6000"
bash ./sizegs.sh garden 4000 2000 2000 0.00003 360_v2 60 20
echo "Completed for name: garden"

sleep 1s

# echo "Running for name: kitchen, iter: 6000"
bash ./sizegs.sh kitchen 4000 2000 2000 0.0005 360_v2 60 10
echo "Completed for name: kitchen"

sleep 1s

# echo "Running for name: room, iter: 6000"
bash ./sizegs.sh room 4000 2000 2000 0.00003 360_v2 60 8
echo "Completed for name: room"

sleep 1s

# echo "Running for name: stump, iter: 6000"
bash ./sizegs.sh stump 4000 2000 2000 0.00003 360_v2 60 24
echo "Completed for name: stump"

sleep 1s

# echo "Running for name: flowers, iter: 6000"
bash ./sizegs.sh flowers 4000 2000 2000 0.0001 360_v2 60 16
echo "Completed for name: flowers"

sleep 1s

# echo "Running for name: treehill, iter: 6000"
bash ./sizegs.sh treehill 4000 2000 2000 0.0001 360_v2 60 18
echo "Completed for name: treehill"

