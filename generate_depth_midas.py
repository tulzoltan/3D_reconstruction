import os
import re
import glob

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from calibration import Calibration


def DownSample_Image(image, reduction_factor):
    """Downsample image reduction_factor number of times"""
    for i in range(0, reduction_factor):
        #check if image is colorful or grayscale
        if len(image.shape) == 3:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize = (col//2, row//2))
    return image


def get_serial_list(npat):
    nums = [re.split('/|_|\.',s)[-2] for s in glob.glob(npat)]
    nums.sort()
    return nums


class DataHandler:
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


def pointcloud_to_file(cloud, file_name):
    ply_header = """ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

    with open(file_name, "w") as f:
        f.write(ply_header %dict(vert_num=len(cloud)))
        np.savetxt(f, cloud, "%f %f %f %d %d %d")


def main():
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")

    main_cam_name = "F_MIDLONGRANGECAM_CL"
    other_cam_names = ["B_MIDRANGECAM_C",
                       "F_MIDRANGECAM_C",
                       "M_FISHEYE_L",
                       "M_FISHEYE_R"]

    #Set up generator
    generator = DataHandler(dat_dir, main_cam_name, other_cam_names)

    #Load MiDaS model for depth estimation
    #model_type = "DPT_Large" # MiDaS v3 - Large
    model_type = "DPT_Hybrid" # MiDaS v3 - hybrid
    #model_type = "MiDaS_small" # MiDaS v2.1 - small
    generator.load_model(model_type)

    count = 0
    for cname in [main_cam_name]:
        out_dir = os.path.join(generator.dep_dir, cname+"_test")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fx = generator.CamCal.Intrinsic_UD[cname][0,0]
        fy = generator.CamCal.Intrinsic_UD[cname][1,1]
        cx = generator.CamCal.Intrinsic_UD[cname][0,2]
        cy = generator.CamCal.Intrinsic_UD[cname][1,2]
        for snum in generator.serials:
            count+=1
            if count < 840:
                continue
            elif count > 850:
                break
            else:
                print(count, snum)
            input_img = generator.load_image(cname, snum)
            depth_map = generator.get_depth_map(input_img)

            #vertices = np.zeros(depth_map.shape+(1,))
            #for i in range(depth_map.shape[0]):
            #    for j in range(depth_map.shape[1]):
            #        z = depth_map[i,j]
            #        x = (j - cx) * z / fx
            #        y = (i - cy) * z / fy
            #        r, g, b = input_img[i,j]
            #        vertices[] = x, y, z, r, g, b
            #vertices = np.array(vertices)

            #pcd_name = cname + "_" + snum + ".ply"
            #pointcloud_to_file(vertices, pcd_name)

            mask1 = depth_map > 30
            input_imp = input_img[mask1]
            depth_map = depth_map*mask1

            img1, img2 = generator.apply_masks(input_img, cname, snum)
            dpt1, dpt2 = generator.apply_masks(depth_map, cname, snum)

            #color_name = cname + "_" + snum + ".jpg"
            #cv2.imwrite(os.path.join(out_dir, color_name), input_img)
            #depth_name = cname + "_" + snum + ".png"
            #cv2.imwrite(os.path.join(out_dir, depth_name), depth_map)

            color_name_1 = cname + "_" + snum + "_env.jpg"
            cv2.imwrite(os.path.join(out_dir, color_name_1), img1)
            color_name_2 = cname + "_" + snum + "_dyn.jpg"
            cv2.imwrite(os.path.join(out_dir, color_name_2), img2)

            depth_name_1 = cname + "_" + snum + "_env.png"
            cv2.imwrite(os.path.join(out_dir, depth_name_1), dpt1)
            depth_name_2 = cname + "_" + snum + "_dyn.png"
            cv2.imwrite(os.path.join(out_dir, depth_name_2), dpt2)

        print(f"Depth maps for {cname} ready")


if __name__ == "__main__":
    main()
