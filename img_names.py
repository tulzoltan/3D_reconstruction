import os
import re
import glob
import json


if __name__ == "__main__":
    dat_dir = os.getcwd() + "/datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor/"
    img_dir = dat_dir + "camera/"
    seg_dir = dat_dir + "camera_seg_bin/"

    main_cam_name = "F_MIDLONGRANGECAM_CL"
    other_cam_names = ["B_MIDRANGECAM_C",
                       "F_MIDRANGECAM_C",
                       "M_FISHEYE_L",
                       "M_FISHEYE_R"]

    def get_list(npat):
        nums = [re.split('/|_|\.',s)[-2] for s in glob.glob(npat)]
        nums.sort()
        return nums

    serials = get_list(img_dir+main_cam_name+"/*.jpg")

    #check for other cameras
    for cname in other_cam_names:
        check = get_list(img_dir+cname+"/*.jpg")
        print(f"same numbers for camera {cname}: {check==serials}")

    #check for segmentation masks
    check_list = get_list(seg_dir+main_cam_name+"/*.png")
    print(f"same numbers for segmentation masks: {check_list==serials}")

    with open("record_serials.json", "w") as mf:
        json.dump(serials, mf)
