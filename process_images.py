import os
import re
import glob
import json
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from calibration import Calibration


def get_serial_list(pattern, n=0):
    nums = [re.split('/|_|\.',s)[-2-n] for s in glob.glob(pattern)]
    nums.sort()
    return nums


def get_egomotion(file_name, serials):
    nol0 = [s.lstrip('0') for s in serials]

    with open(file_name, "r") as file:
        data = json.load(file)

    short = { snum: np.array(data[s]["RT_ECEF_body"]) for s, snum in zip(nol0, serials) }

    return short


def get_rays_cam(width, height):
    ones = np.ones((height, width))

    #ray directions in camera coordinate system
    u, v = np.meshgrid(np.arange(width),
                       np.arange(height))
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    dz = ones

    #normalize
    dnorm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= dnorm
    dy /= dnorm
    dz /= dnorm

    ray_dirs_cam = np.dstack([dx, dy, dz])

    return ray_dirs_cam, ones


def DownSample_Image(image, reduction_factor):
    """Downsample image reduction_factor number of times"""
    for _ in range(reduction_factor):
        #check if image is colorful or grayscale
        if len(image.shape) == 3:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize = (col//2, row//2))
    return image


if __name__ == "__main__":
    #Path components
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")

    img_dir = os.path.join(dat_dir, "camera")
    cal_dir = os.path.join(dat_dir, "calibration")
    ego_dir = os.path.join(dat_dir, "gnssins")

    camera_name = "F_MIDLONGRANGECAM_CL"

    #Get intrinsic and extrinsic matrices
    CamCal = Calibration(cal_dir, "calibration.json", [camera_name])


    W = CamCal.width[camera_name]
    H = CamCal.height[camera_name]

    #reduce for downsampled images
    red_fac = 1
    for _ in range(red_fac):
        H = H // 2
        W = W // 2

    fx = CamCal.Intrinsic_UD[camera_name][0,0]
    fy = CamCal.Intrinsic_UD[camera_name][1,1]
    cx = CamCal.Intrinsic_UD[camera_name][0,2]
    cy = CamCal.Intrinsic_UD[camera_name][1,2]
    extrinsic = CamCal.Extrinsic[camera_name]

    #Get serial numbers
    serials = get_serial_list(
            os.path.join(img_dir, camera_name, "*.jpg"), 0)

    #Get egomotion data
    trajectory = get_egomotion(
            os.path.join(ego_dir, "egomotion2.json"), serials)

    ray_dirs_cam, ones = get_rays_cam(width=W, height=H)

    ext_base = np.linalg.inv(trajectory[serials[0]])

    chunks = 20
    jump = 5
    chunk_size = len(serials) // chunks
    for chunk_ind in range(chunks):
        if chunk_ind != chunks-2:
            continue
        lower = chunk_ind * chunk_size
        upper = (chunk_ind + 1) * chunk_size

        ds_train = np.empty((chunk_size//jump*H*W, 9),
                             dtype=np.float32)

        img_ind = 0
        for sub_ind in range(lower, upper, jump):
            snum = serials[sub_ind]
            ext = ext_base @ trajectory[snum] @ extrinsic
            R = ext[:3, :3]
            t = ext[:3, 3]
        
            img_name = camera_name + "_" + snum + ".jpg"
            img = cv2.imread(
                    os.path.join(img_dir, camera_name, img_name))
            img = CamCal.undistort(img, camera_name)

            img = DownSample_Image(img, red_fac)

            #ray directions
            ray_dirs = np.einsum('ij,klj->kli', R, ray_dirs_cam)

            #ray origins
            ray_oris = np.dstack([t[0]*ones, t[1]*ones, t[2]*ones])

            #combine
            pixels = np.hstack([ray_oris.reshape(-1, 3),
                                ray_dirs.reshape(-1, 3),
                                img.reshape(-1, 3)])

            ds_train[img_ind: img_ind + H*W] = pixels
            img_ind += H*W

        output_name = "pixdat_" + camera_name + "_" + str(chunk_ind) + ".pkl"
        with open(output_name, "wb") as file:
            pickle.dump(ds_train, file)
