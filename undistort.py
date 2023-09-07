import json
import numpy as np
import cv2 as cv


def get_calibration_data(calibration_file, camera_names):
    """Get intrinsic matrix and distortion parameters from calibration file"""

    CameraInfo = {}

    with open(calibration_file) as in_file:
        data = json.load(in_file)
        for cname in camera_names:
            cdata = data[cname]

            #Construct intrinsic matrix
            Intrinsic = np.eye(3)
            Intrinsic[0,0], Intrinsic[1,1] = cdata["focal_length_px"]
            Intrinsic[0,2], Intrinsic[1,2] = cdata["principal_point_px"]

            Distortion = np.array(cdata["distortion_coeffs"])
            CameraInfo[cname] = {
                 "intrinsic": Intrinsic,
                 "distortion": Distortion,
                 }

    return CameraInfo


def undist(img_in, CameraMatrix, Dist):
    """Undistort and crop image"""
    h, w = img_in.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(CameraMatrix, Dist, (w,h), 1, (w,h))

    #Undistort
    #img_out = cv.undistort(img_in, CameraMatrix, Dist, None, newCameraMatrix)

    #Undistort using remap
    mapx, mapy = cv.initUndistortRectifyMap(CameraMatrix, Dist, None, newCameraMatrix, (w,h), 5)
    img_out = cv.remap(img_in, mapx, mapy,
                       interpolation=cv.INTER_LINEAR,
                       borderMode=cv.BORDER_CONSTANT)

    # crop image
    x, y, w, h = roi
    img_out = img_out[y:y+h, x:x+w]

    return img_out


def undist_fe(img_in, CameraMatrix, Dist):
    """Undistort and crop image for fisheye cameras"""
    h, w = img_in.shape[:2]
    #newCameraMatrix = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(CameraMatrix, Dist[:-1], (w,h), np.eye(3))

    #Undistort
    mapx, mapy = cv.fisheye.initUndistortRectifyMap(CameraMatrix, Dist[:-1], None, CameraMatrix, (w,h), cv.CV_16SC2) #or cv.CV32FC1
    img_out = cv.remap(img_in, mapx, mapy,
                       interpolation=cv.INTER_LINEAR,
                       borderMode=cv.BORDER_CONSTANT)

    return img_out


if __name__ == "__main__":
    import os

    #Path components
    dat_dir = os.getcwd() + "/datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor/"
    camera_names = ["B_MIDRANGECAM_C",
                    "F_MIDLONGRANGECAM_CL",
                    "F_MIDRANGECAM_C",
                    "M_FISHEYE_L",
                    "M_FISHEYE_R"]
    camera_names_1 = ["B_MIDRANGECAM_C", "F_MIDLONGRANGECAM_CL",
                      "F_MIDRANGECAM_C"]
    camera_names_2 = ["M_FISHEYE_L", "M_FISHEYE_R"]
    img_dir = dat_dir + "camera/"
    cal_dir = dat_dir + "calibration/"

    #get intrinsic matrix and distortion parameters
    CameraInfo = get_calibration_data(cal_dir+"calibration.json",
                                      camera_names)

    #Load and undistort images
    for cname in camera_names:
        img_file = img_dir + cname + "/" + cname + "_0037133.jpg"
        img = cv.imread(img_file)

        CameraMatrix = CameraInfo[cname]["intrinsic"]
        Dist         = CameraInfo[cname]["distortion"]
        if cname in camera_names_1:
            dst = undist(img, CameraMatrix, Dist)
            cv.imwrite(os.getcwd()+"/"+cname+"_1.png", dst)
        elif cname in camera_names_2:
            dst = undist_fe(img, CameraMatrix, Dist)
            cv.imwrite(os.getcwd()+"/"+cname+"_2.png", dst)

        cv.imwrite(os.getcwd()+"/"+cname+"_0.png", img)

