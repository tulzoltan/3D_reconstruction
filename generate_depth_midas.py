import os
import re
import glob

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from calibration import Calibration


def get_serial_list(pattern, n=0):
    nums = [re.split('/|_|\.',s)[-2-n] for s in glob.glob(pattern)]
    nums.sort()
    return nums


class DepthEstimator_midas:
    def __init__(self, data_dir, main_camera, other_cameras):
        assert os.path.exists(data_dir)

        self.img_dir = os.path.join(data_dir, "camera")
        self.seg_dir = os.path.join(data_dir, "camera_seg_bin")
        self.cal_dir = os.path.join(data_dir, "calibration")

        self.dep_dir = os.path.join(os.getcwd(), "processed_images")
        if not os.path.exists(self.dep_dir):
            os.mkdir(self.dep_dir)

        self.main_camera = main_camera
        self.other_cameras = other_cameras
        self.cameras = [self.main_camera, ] + other_cameras
        self.cameras.sort()

        #Get serial numbers for image files
        self.serials = get_serial_list(
                os.path.join(self.img_dir, self.main_camera, "*.jpg"))
        print(f"File serial numbers obtained for {self.main_camera}")

        for cname in self.other_cameras:
            check = get_serial_list(
                    os.path.join(self.img_dir, cname, "*.jpg"))
            print(f"Serial numbers match for {cname}: {check==self.serials}")

        check = get_serial_list(
                os.path.join(self.seg_dir, self.main_camera, "*.png"))
        print(f"Serial numbers match for segmentation masks: {check==self.serials}")

        #Get calibration data
        self.CamCal = Calibration(
                        self.cal_dir, "calibration.json", self.cameras)

    def load_model(self, model_type):
        if not model_type in ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]:
            import sys
            sys.exit("Invalid model name")
        
        #Load model
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        #Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        #Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def load_image(self, camera_name, serial_number):
        #Read image file
        img_name = camera_name + "_" + serial_number + ".jpg"
        img = cv2.imread(
                os.path.join(self.img_dir, camera_name, img_name))
        #Undistort image
        img = self.CamCal.undistort(img, camera_name)
        return img

    def apply_masks(self, image, camera_name, serial_number):
        #Segmentation mask
        seg_name = camera_name + "_" + serial_number + ".png"
        seg = cv2.imread(
                os.path.join(self.seg_dir, camera_name, seg_name))
        seg = self.CamCal.undistort(seg, camera_name)
        #Obstruction mask
        obs = self.CamCal.ObstructionMask[camera_name] > 0
        if len(image.shape)==2:
            seg = seg[:,:,0]
            obs = obs[:,:,0]
        dif1 = obs*seg*image
        dif2 = obs*(1-seg)*image
        return dif1, dif2

    def get_depth_map(self, image):
        #Transform image
        image_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).to(self.device)

        #Predict depth map
        with torch.no_grad():
            prediction = self.midas(image)

            prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=image_size,
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 1,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        depth_map = (depth_map*255).astype(np.uint8)

        return depth_map


def main():
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")

    main_cam_name = "F_MIDLONGRANGECAM_CL"
    other_cam_names = ["B_MIDRANGECAM_C",
                       "F_MIDRANGECAM_C",
                       "M_FISHEYE_L",
                       "M_FISHEYE_R"]
    other_cam_names = ["B_MIDRANGECAM_C", "F_MIDRANGECAM_C"]

    #Set up generator
    generator = DepthEstimator_midas(dat_dir, main_cam_name, other_cam_names)

    #Load MiDaS model for depth estimation
    model_type = "DPT_Large" # MiDaS v3 - Large
    #model_type = "DPT_Hybrid" # MiDaS v3 - hybrid
    #model_type = "MiDaS_small" # MiDaS v2.1 - small
    generator.load_model(model_type)

    count = 0
    for cname in [main_cam_name]:
        out_dir = os.path.join(generator.dep_dir, cname)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for snum in generator.serials:
            count+=1
            if count < 800: #840 - 850
                continue
            elif count > 850:
                break
            else:
                print(count, snum)
            input_img = generator.load_image(cname, snum)
            depth_map = generator.get_depth_map(input_img)

            mask1 = depth_map > 30
            input_img = input_img*np.repeat(mask1[:,:,np.newaxis],3,axis=2)
            depth_map = depth_map*mask1

            if cname == main_cam_name:
                img1, img2 = generator.apply_masks(input_img, cname, snum)
                dpt1, dpt2 = generator.apply_masks(depth_map, cname, snum)

                #Print color maps to file
                color_name_1 = cname + "_" + snum + "_env.jpg"
                cv2.imwrite(os.path.join(out_dir, color_name_1), img1)
                color_name_2 = cname + "_" + snum + "_dyn.jpg"
                cv2.imwrite(os.path.join(out_dir, color_name_2), img2)

                #Print depth maps to file
                depth_name_1 = cname + "_" + snum + "_env.png"
                cv2.imwrite(os.path.join(out_dir, depth_name_1), dpt1)
                depth_name_2 = cname + "_" + snum + "_dyn.png"
                cv2.imwrite(os.path.join(out_dir, depth_name_2), dpt2)

            else:
                #Print color maps to file
                color_name = cname + "_" + snum + ".jpg"
                cv2.imwrite(os.path.join(out_dir, color_name), input_img)

                #Print depth map to file
                depth_name = cname + "_" + snum + ".png"
                cv2.imwrite(os.path.join(out_dir, depth_name), depth_map)

        print(f"Depth maps for {cname} ready")


if __name__ == "__main__":
    main()
