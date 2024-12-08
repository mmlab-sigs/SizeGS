# !/bin/bash


echo "Running for name: bicycle, iter: 6000"
bash ./meson.sh bicycle  8000 1000 0.0001 360_v2 60 12
echo "Completed for name: bicycle"

sleep 20s

echo "Running for name: bonsai, iter: 6000"
bash ./meson.sh bonsai  6000 300 0.00008 360_v2 60 6 
echo "Completed for name: bonsai"

sleep 20s

echo "Running for name: counter, iter: 6000"
bash ./meson.sh counter  6000 300 0.00003 360_v2 60 4
echo "Completed for name: counter"

sleep 20s

echo "Running for name: garden, iter: 6000"
bash ./meson.sh garden  6000 300 0.00003 360_v2 60 10 
echo "Completed for name: garden"

sleep 20s

echo "Running for name: kitchen, iter: 6000"
bash ./meson.sh kitchen  6000 300 0.0005 360_v2 60 5
echo "Completed for name: kitchen"

sleep 20s

echo "Running for name: room, iter: 6000"
bash ./meson.sh room  6000  300 0.00003 360_v2 60 3 
echo "Completed for name: room"

sleep 20s

echo "Running for name: stump, iter: 6000"
bash ./meson.sh stump  6000 300 0.00003 360_v2 60 12
echo "Completed for name: stump"

sleep 20s

echo "Running for name: flowers, iter: 6000"
bash ./meson.sh flowers  6000 300 0.0001 360_v2 60 8 
echo "Completed for name: flowers"

sleep 20s

echo "Running for name: treehill, iter: 6000"
bash ./meson.sh treehill  6000 300 0.0001 360_v2 60 9
echo "Completed for name: treehill"




