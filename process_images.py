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

    short = { snum: {"RT": np.array(data[s]["RT_ECEF_body"]), "time": data[s]["time_host"]} for s, snum in zip(nol0, serials) }

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
    def __init__(self, width, height, fx, fy, cx, cy):
        #ray directions in camera coordinate system
        u, v = np.meshgrid(np.arange(width),
                       np.arange(height))
        dx = (u - cx) / fx
        dy = (v - cy) / fy
        dz = np.ones_like(dx)

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

        return ray_oris, ray_dirs


def load_image(img_dir, camera_name, snum, CamCal, reduction_factor=0):
    #load
    img_name = camera_name + "_" + snum + ".jpg"
    img = cv2.imread(
            os.path.join(img_dir, camera_name, img_name))

    #undistort
    img = CamCal.undistort(img, camera_name)

    #crop obstruction
    img = CamCal.crop_image(img, camera_name)

    #downsample
    img = DownSample_Image(img, reduction_factor)

    #convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img


if __name__ == "__main__":
    #Path components
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")

    img_dir = os.path.join(dat_dir, "camera")
    cal_dir = os.path.join(dat_dir, "calibration")
    ego_dir = os.path.join(dat_dir, "gnssins")

    meta_file = os.path.join(os.getcwd(), "metadata.json")
    meta_dict = {}

    camera_names = ["F_MIDLONGRANGECAM_CL",
                    "F_MIDRANGECAM_C"]
    #camera_names = ["F_MIDLONGRANGECAM_CL"]

    #Get intrinsic and extrinsic matrices
    CamCal = Calibration(cal_dir, "calibration.json", camera_names)

    #Get serial numbers
    serials = get_serial_list(
            os.path.join(img_dir, camera_names[0], "*.jpg"))
    print(f"File serial numbers obtained for {camera_names[0]}")

    for i in range(1,len(camera_names)):
        check = get_serial_list(
                os.path.join(img_dir, camera_names[i], "*.jpg"))
        print(f"Serial numbers match for {camera_names[i]}: {(check==serials).all()}")

    #Get egomotion data
    trajectory = get_egomotion(
            os.path.join(ego_dir, "egomotion2.json"), serials)

    #matrix for setting initial position
    ext_base = np.linalg.inv(trajectory[serials[0]]["RT"])

    #reduction factor for downsampling images, 0 is no downsampling
    red_fac = 2

    #split images into chunks and iterate through them in jumps
    chunks = 20
    jump = 1
    chunk_size = len(serials) // chunks


    #loop over cameras
    for camera_name in camera_names:
        print(f"processing images for {camera_name} ...")
        H, W = CamCal.get_cropped_height_width(camera_name)
        intrinsic = CamCal.get_intrinsic_crop(camera_name)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        extrinsic = CamCal.Extrinsic[camera_name]

        #reduce for downsampled images
        for _ in range(red_fac):
            H = H // 2
            W = W // 2
            fx = fx // 2
            fy = fy // 2
            cx = cx // 2
            cy = cy // 2

        #Make rays
        rays = RayMaker(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)

        output_file_list = []
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
                ext = ext_base @ trajectory[snum]["RT"] @ extrinsic

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

                dataset[img_ind_1*H*W: (img_ind_1+1)*H*W] = pixels
                img_ind_1 += 1

            output_name = "pixdat_" + camera_name + "_" + str(chunk_ind) + ".pkl"
            output_file_list.append(output_name)
            with open(output_name, "wb") as file:
                pickle.dump(dataset, file)

            print(f"{img_ind_1} images saved to {output_name}")
            print(f"number of pixels: {len(dataset)}")

        meta_dict[camera_name] = {
                            "image_height": H,
                            "image_width": W,
                            "focal_x": fx,
                            "focal_y": fy,
                            "reduction_factor": red_fac,
                            "file_names": output_file_list}

    with open(meta_file, "w") as mf:
        json.dump(meta_dict, mf)
