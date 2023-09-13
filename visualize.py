import numpy as np
import open3d as o3d
import os

from calibration import Calibration

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.1, 0.1, 0.1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == "__main__":
    #Path components
    cal_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor/calibration")
    camera_name = "F_MIDLONGRANGECAM_CL"

    #Get intrinsic matrix
    CamCal = Calibration(cal_dir, "calibration.json", [camera_name])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=CamCal.width[camera_name],
            height=CamCal.height[camera_name],
            intrinsic_matrix=CamCal.Intrinsic_UD[camera_name],
            )

    extrinsic = CamCal.Extrinsic[camera_name]

    serial = "0037532"
    image_name = camera_name+"_"+serial+"_env"
    image_path = os.path.join(os.getcwd(), "processed_images", camera_name+"_test", image_name)

    #Load color and depth images
    color_raw = o3d.io.read_image(image_path+".jpg")
    depth_raw = o3d.io.read_image(image_path+".png")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, 1000.0, 3.0, False
            )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic
            )
    
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])

    cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
    pcd_inlier=pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_inlier])
