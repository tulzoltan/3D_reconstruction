import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import undistort as ud


if __name__ == "__main__":
    import os

    #Path components
    dat_dir = os.getcwd() + "/datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor/"
    camera_name = "F_MIDLONGRANGECAM_CL"
    img_dir = dat_dir + "camera/" + camera_name + "/"


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

    #Load image for test
    img_name = camera_name + "_0037498"
    img = cv2.imread(img_dir+img_name+".jpg")
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

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title("Depth")
    plt.imshow(depth_map)
    plt.show()

    #cv2.imshow("Image", img)
    #cv2.imshow("Depth", depth_map)
    #cv2.waitKey(0)
