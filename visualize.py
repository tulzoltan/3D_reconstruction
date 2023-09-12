import numpy as np
import open3d as o3d

from undistort import Calibration

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.1, 0.1, 0.1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])





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

    extrinsic = CamCal.Extrinsic[camera_name]

    serial = "0037532"
    image_name = camera_name+"_"+serial
    image_path = os.path.join(os.getcwd(), "processed_images", camera_name+"_test", image_name)

    #Load color and depth images
    color_raw = o3d.io.read_image(image_path+".jpg")
    depth_raw = o3d.io.read_image(image_path+".png")

    depth_raw = np.asarray(depth_raw)
    mask = depth_raw > 30.0
    #color_raw = np.asarray(color_raw)

    depth_raw = o3d.geometry.Image(depth_raw*mask)
    #mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    #color_raw = o3d.geometry.Image(color_raw*mask)

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
