bash ./test_room.sh bicycle 6000 0 0 0.001 360_v2 60 24 1
bash ./test_room.sh bicycle 5000 1000 0 0.001 360_v2 60 24 1
bash ./test_room.sh bicycle 5000 0 1000 0.001 360_v2 60 24 1
bash ./test_room.sh bicycle 4000 1000 1000 0.001 360_v2 60 24 1


bash ./render.sh bicycle 360_v2 6000 outputs/360_v2/bicycle/aba_iterations/2025-04-12_03:58:23
bash ./render.sh bicycle 360_v2 4400 outputs/360_v2/bicycle/aba_iterations/2025-04-12_04:30:01

bash ./render.sh bicycle 360_v2 4000 outputs/360_v2/bicycle/aba_iterations/2025-04-12_05:03:19