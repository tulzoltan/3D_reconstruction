import os
import re
import glob

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from undistort import Calibration


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


if __name__ == "__main__":
    #Path components
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")
    assert os.path.exists(dat_dir)

    main_cam_name = "F_MIDLONGRANGECAM_CL"
    other_cam_names = ["B_MIDRANGECAM_C",
                       "F_MIDRANGECAM_C",
                       "M_FISHEYE_L",
                       "M_FISHEYE_R"]

    img_dir = os.path.join(dat_dir, "camera")
    seg_dir = os.path.join(dat_dir, "camera_seg_bin")
    cal_dir = os.path.join(dat_dir, "calibration")

    dep_dir = os.path.join(os.getcwd(), "depth_maps")
    if not os.path.exists(dep_dir):
        os.mkdir(dep_dir)


    #Get serial numbers for image files
    serials = get_serial_list(os.path.join(img_dir, main_cam_name, "*.jpg"))
    print(f"File serial numbers obtained for {main_cam_name}")

    for cname in other_cam_names:
        check = get_serial_list(os.path.join(img_dir, cname, "*.jpg"))
        print(f"Serial numbers match for {cname}: {check==serials}")

    check = get_serial_list(os.path.join(seg_dir, main_cam_name, "*.png"))
    print(f"Serial numbers match for segmentation masks: {check==serials}")


    #Load MiDaS model for depth estimation
    #model_type = "DPT_Large" # MiDaS v3 - Large
    model_type = "DPT_Hybrid" # MiDaS v3 - hybrid
    #model_type = "MiDaS_small" # MiDaS v2.1 - small

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    #move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    #Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    #Calibration data
    CamCal = Calibration(os.path.join(cal_dir, "calibration.json"), [main_cam_name])

    #Load image for test
    for cname in [main_cam_name]:
        out_dir = os.path.join(dep_dir, cname)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for snum in serials:
            img_name = cname + "_" + snum + ".jpg"
            img = cv2.imread(os.path.join(img_dir, cname, img_name))
            img = CamCal.undist(img, cname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_img = transform(img).to(device)

            #Prediction
            with torch.no_grad():
                prediction = midas(input_img)

                prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()

            depth_map = prediction.cpu().numpy()

            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

            depth_map = (depth_map*255).astype(np.uint8)

            #plt.subplot(1,2,1)
            #plt.title("Image")
            #plt.imshow(img)
            #plt.subplot(1,2,2)
            #plt.title("Depth")
            #plt.imshow(cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA))
            #plt.show()

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_name = cname + "_" + snum + ".png"
            cv2.imwrite(os.path.join(out_dir, out_name), depth_map)

        print(f"Depth maps for {cname} ready")

