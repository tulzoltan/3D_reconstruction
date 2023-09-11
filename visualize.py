import numpy as np
import open3d as o3d

from undistort import Calibration


if __name__ == "__main__":
    import os

    #Path components
    dat_dir = os.getcwd() + "/datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor/"
    camera_name = "F_MIDLONGRANGECAM_CL"
    cal_dir = dat_dir + "calibration/"

    #Get intrinsic matrix
    CamCal = Calibration(cal_dir+"calibration.json", [camera_name])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1819,
            height=955,
            intrinsic_matrix=CamCal.Intrinsic[camera_name],
            )

    #Load color and depth images
    color_raw = o3d.io.read_image("Image.jpg")
    depth_raw = o3d.io.read_image("Depth.png")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw
            )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
            )
    
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    o3d.visualization.draw_geometries([pcd])
