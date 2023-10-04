# 3D_reconstruction

Set up the conda environment using the .yaml file
  ```shell
  conda env create --file conda_environment_ptcv.yaml
  ```

Provide the path to sensor data using the variable dat_dir in the file process_images.py and set hyperparameters in the file nerf_model.py


Process image files using 
  ```shell
  python process_iamges.py
  ```

Train neural network
  ```shell
  python nerf_model.py
  ```
Check training loss after 100 steps to verify convergence. If no convergence is observed, restart training.

Evaluate neural network
  ```shell
  python nerf_test.py
  ```
