# 3D_reconstruction

Set up the conda environment using the yaml file

[] conda env create --file conda_environment_ptcv.yaml


Provide the path to sensor data using the variable dat_dir in main() in the file generate_depth_midas.py


Generate depth maps

[] python generate_depth_midas.py


Create and display point cloud

[] python visualize.py
