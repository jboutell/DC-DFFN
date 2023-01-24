# DC-DFFN
DC-DFFN: Densely Connected Deep Feature Fusion Network with Sign Agnostic Learning for Implicit Shape Representation


# Generation 

cd ./code  

python evaluate/eval.py --expname shapenet --parallel --exps_folder_name trained_models --timestamp 2022_08_19_16_19_30 --checkpoints 1500 --confs ./confs/shapenet_vae.conf --split ./confs/splits/shapenet/shapenet_sofa_test_files.conf --resolution 100



# Acknowledgemetn 
This code is based on SALD (https://github.com/matanatz/SALD), thanks for this wonderful work. 
