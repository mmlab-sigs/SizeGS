#!/bin/bash
echo "Running for name: drjohnson, iter: 8000"
bash ./meson.sh drjohnson 8000 300 0.0001 db 60 10
echo "Completed for name: materials"

sleep 20s

echo "Running for name: playroom, iter: 8000"
bash ./meson.sh playroom 8000 300 0.0001 db 60 10
echo "Completed for name: mic"