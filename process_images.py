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
    return np.array(nums)


def get_egomotion(file_name, serials):
    nol0 = [s.lstrip('0') for s in serials]

    with open(file_name, "r") as file:
        data = json.load(file)

    short = { snum: np.array(data[s]["RT_ECEF_body"]) for s, snum in zip(nol0, serials) }

    return short


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


class RayMaker():
    def __init__(self, width, height, fx, fy):
        #self.ones = np.ones((height, width))

        #ray directions in camera coordinate system
        u, v = np.meshgrid(np.arange(width),
                       np.arange(height))
        dx = (u - width/2.) / fx
        dy = (v - height/2.) / fy
        dz = np.ones_like(dx)
        #dz = self.ones

        #normalize
        dnorm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx /= dnorm
        dy /= dnorm
        dz /= dnorm

        self.ray_dirs_cam = np.dstack([dx, dy, dz])

    def make(self, extrinsic):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        #ray directions
        ray_dirs = np.einsum('ij,klj->kli', R,
                             self.ray_dirs_cam)

        #ray origins
        ray_oris = np.broadcast_to(t, ray_dirs.shape)
        #ray_oris = np.dstack([t[0]*self.ones,
        #                      t[1]*self.ones,
        #                      t[2]*self.ones])

        return ray_oris, ray_dirs


def load_image(img_dir, camera_name, snum, CamCal, reduction_factor=0):
    img_name = camera_name + "_" + snum + ".jpg"
    img = cv2.imread(
            os.path.join(img_dir, camera_name, img_name))
    img = CamCal.undistort(img, camera_name)
    img = DownSample_Image(img, reduction_factor)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


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

    intrinsic = CamCal.Intrinsic_UD[camera_name]
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    #cx = intrinsic[0,2]
    #cy = intrinsic[1,2]
    extrinsic = CamCal.Extrinsic[camera_name]

    #reduce for downsampled images
    red_fac = 2
    for _ in range(red_fac):
        H = H // 2
        W = W // 2
        fx = fx // 2
        fy = fy // 2
    print("image size (H, W): ", H, W)

    #Get serial numbers
    serials = get_serial_list(
            os.path.join(img_dir, camera_name, "*.jpg"), 0)

    #Get egomotion data
    trajectory = get_egomotion(
            os.path.join(ego_dir, "egomotion2.json"), serials)

    #Make rays
    rays = RayMaker(width=W, height=H, fx=fx, fy=fy)

    ext_base = np.linalg.inv(trajectory[serials[0]])

    chunks = 20
    jump = 1 #3, 5
    chunk_size = len(serials) // chunks
    for chunk_ind in range(chunks):
        if chunk_ind < 3: #chunks-3
            continue
        elif chunk_ind > 4: #chunks-2
            break
        lower = chunk_ind * chunk_size
        upper = (chunk_ind + 1) * chunk_size

        dataset = np.empty((chunk_size//jump*H*W, 9),
                            dtype=np.float32)

        img_ind_1 = 0
        for sub_ind in range(lower, upper, jump):
            snum = serials[sub_ind]
            ext = ext_base @ trajectory[snum] @ extrinsic

            #get ray origins and directions
            ray_oris, ray_dirs = rays.make(ext)

            #load image
            img = load_image(img_dir, camera_name,
                             snum, CamCal, 
                             reduction_factor=red_fac)

            #combine
            pixels = np.hstack([ray_oris.reshape(-1, 3),
                                ray_dirs.reshape(-1, 3),
                                img.reshape(-1, 3)])

            dataset[img_ind_1*H*W: (img_ind_1 + 1)*H*W] = pixels
            img_ind_1 += 1

        output_name = "pixdat_" + camera_name + "_" + str(chunk_ind) + ".pkl"
        with open(output_name, "wb") as file:
            pickle.dump(dataset, file)

        print(f"{img_ind_1} images saved to {output_name}")
        print(f"number of pixels: {len(dataset)}")
