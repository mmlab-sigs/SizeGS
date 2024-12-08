echo "Running for name: train, iter: 6000"
bash ./meson.sh train 6000 300 0.0002 tandt 60 10
echo "Completed for name: train"

sleep 20s

echo "Running for name: truck, iter: 6000"
bash ./meson.sh truck 6000 300 0.00008 tandt 60 12
echo "Completed for name: truck"
