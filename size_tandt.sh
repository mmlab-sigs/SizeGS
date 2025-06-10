#!/bin/bash
# echo "Running for name: drjohnson, iter: 8000"
bash ./sizegs.sh train 6000 4000 4000 0.001 tandt 100 10 0
# echo "Completed for name: drjohnson"

sleep 1s

# echo "Running for name: playroom, iter: 8000"
bash ./sizegs.sh truck 6000 4000 4000 0.001 tandt 100 12 0
# echo "Completed for name: playroom"
