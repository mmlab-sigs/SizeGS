# !/bin/bash


echo "Running for name: chair, iter: 6000"
bash ./meson.sh chair  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: chair"

sleep 20s

echo "Running for name: drums, iter: 6000"
bash ./meson.sh drums  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: drums"

sleep 20s

echo "Running for name: ficus, iter: 6000"
bash ./meson.sh ficus  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: ficus"

sleep 20s

echo "Running for name: hotdog, iter: 6000"
bash ./meson.sh hotdog  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: hotdog"

sleep 20s

echo "Running for name: lego, iter: 6000"
bash ./meson.sh lego  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: lego"

sleep 20s

echo "Running for name: materials, iter: 6000"
bash ./meson.sh materials  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: materials"

sleep 20s

echo "Running for name: mic, iter: 6000"
bash ./meson.sh mic  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: mic"

sleep 20s

echo "Running for name: ship, iter: 6000"
bash ./meson.sh ship  6000 1000 0.0001 nerf_synthetic 60 1
echo "Completed for name: ship"



