# 3D_reconstruction

Set up the conda environment using the .yaml file
  ```shell
  conda env create --file conda_environment_ptcv.yaml
  ```

Provide the path to sensor data using the variable dat_dir in main() in the file generate_depth_midas.py


Generate depth maps
  ```shell
  python generate_depth_midas.py
  ```

Create and display point cloud
  ```shell
  python visualize.py
  ```
