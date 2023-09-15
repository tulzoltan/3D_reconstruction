import os
import json
import numpy as np
import open3d as o3d

from calibration import Calibration
from generate_depth_midas import get_serial_list


def mswitch(M, i, j):
    M[[i,j],:] = M[[j,i],:]
    M[:,[i,j]] = M[:,[j,i]]
    return M


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.1, 0.1, 0.1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def get_egomotion(file_name, serials):
    nol0 = [s.lstrip('0') for s in serials]

    with open(file_name, "r") as f:
        data = json.load(f)

    short = { snum: np.array(data[s]["RT_ECEF_body"]) for s, snum in zip(nol0, serials) }

    return short


if __name__ == "__main__":
    #Path components
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")
    cal_dir = os.path.join(dat_dir, "calibration")
    ego_dir = os.path.join(dat_dir, "gnssins")

    camera_name = "F_MIDLONGRANGECAM_CL"

    #Get intrinsic and extrinsic matrices
    CamCal = Calibration(cal_dir, "calibration.json", [camera_name])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=CamCal.width[camera_name],
            height=CamCal.height[camera_name],
            intrinsic_matrix=CamCal.Intrinsic_UD[camera_name],
            )

    extrinsic = CamCal.Extrinsic[camera_name]

    #Image path
    img_path = os.path.join(os.getcwd(), "processed_images", camera_name)

    #Get serial numbers
    serials = get_serial_list(
            os.path.join(os.getcwd(),img_path,"*_env.jpg"), 1)

    #Get egomotion data
    trajectory = get_egomotion(
            os.path.join(ego_dir, "egomotion2.json"), serials)

    pcds = []
    counter = 0
    ext_base = None
    for snum in serials:
        counter += 1
        if counter < 40:
            continue
        if counter > 41:
            break
        if counter == 40:
            ext_base = np.linalg.inv(trajectory[snum])
        #serial = "0037532"
        img_name = camera_name+"_"+snum+"_env"

        #Load color and depth images
        color_raw = o3d.io.read_image(
                os.path.join(img_path, img_name+".jpg"))
        depth_raw = o3d.io.read_image(
                os.path.join(img_path, img_name+".png"))

        scale = 0.0022 #using info from basler website

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, scale, 3000./scale, False
            )

        world_ext = ext_base @ trajectory[snum]
        #world_ext = mswitch(world_ext,2,0)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, world_ext @ extrinsic,
            #rgbd_image, intrinsic, extrinsic,
            #rgbd_image, intrinsic, world_ext,
            )
 
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])

        #pcd = pcd.voxel_down_sample(voxel_size=2.)
        pcd = pcd.uniform_down_sample(every_k_points=5)

        cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=10, std_ratio=1.0)
        pcd_inlier=pcd.select_by_index(ind)
        #o3d.visualization.draw_geometries([pcd_inlier])

        pcds.append(pcd_inlier)

    o3d.visualization.draw_geometries(pcds)
