import json
import numpy as np
import cv2 as cv


def get_masks(calibration_file, camera_names):
    """Get file names for misalignment and obstruction masks."""
    mask_files = {}

    with open(calibration_file) as in_file:
        data = json.load(in_file)
        for cname in camera_names:
            cdata=data[cname]["custom_vars"]
            mask_files[cname] = {
                    "mis_mask": cdata["misalign_mask_file"],
                    "mis_ref":  cdata["misalign_ref_file"],
                    "obs_mask": cdata["obstruction_mask_file"],
                    }

    return mask_files



class Calibration():
    def __init__(self, calibration_file, camera_names):
        """Get intrinsic matrix and distortion parameters from calibration file"""
        self.Intrinsic = {}
        self.Distortion = {}
        self.Extrinsic = {}
        self.ObstructionMask = {}

        with open(calibration_file) as in_file:
            data = json.load(in_file)
            for cname in camera_names:
                cdata = data[cname]

                #Intrinsic matrix
                self.Intrinsic[cname] = np.eye(3)
                self.Intrinsic[cname][0,0], self.Intrinsic[cname][1,1] = cdata["focal_length_px"]
                self.Intrinsic[cname][0,2], self.Intrinsic[cname][1,2] = cdata["principal_point_px"]

                #Distortion parameters
                self.Distortion[cname] = np.array(cdata["distortion_coeffs"])

                #Extrinsic matrix
                self.Extrinsic[cname] = np.array(cdata["RT_sensor_from_body"])

                #Obstruction mask
                self.ObstructionMask[cname] = cdata["custom_vars"]["obstruction_mask_file"]


    def undistort(self, img_in, cam_name):
        """Undistort and crop image, handle fisheye cameras separately"""
        h, w = img_in.shape[:2]
        CamMat = self.Intrinsic[cam_name]

        if "FISHEYE" in cam_name:
            Dist = self.Distortion[cam_name][:-1]
            #newCamMat = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(CamMat, Dist, (w,h), np.eye(3))
            #Undistort
            mapx, mapy = cv.fisheye.initUndistortRectifyMap(CamMat, Dist, None, CamMat, (w,h), cv.CV_16SC2) #or cv.CV_32FC1
            img_out = cv.remap(img_in, mapx, mapy,
                       interpolation=cv.INTER_LINEAR,
                       borderMode=cv.BORDER_CONSTANT)

        else:
            Dist = self.Distortion[cam_name]
            newCamMat, roi = cv.getOptimalNewCameraMatrix(CamMat, Dist, (w,h), 1, (w,h))

            #Undistort
            #img_out = cv.undistort(img_in, CamMat, Dist, None, newCamMat)

            #Undistort using remap
            mapx, mapy = cv.initUndistortRectifyMap(CamMat, Dist, None, newCamMat, (w,h), 5)
            img_out = cv.remap(img_in, mapx, mapy,
                       interpolation=cv.INTER_LINEAR,
                       borderMode=cv.BORDER_CONSTANT)

            # crop image
            x, y, w, h = roi
            img_out = img_out[y:y+h, x:x+w]

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
    img_dir = dat_dir + "camera/"
    cal_dir = dat_dir + "calibration/"

    #get intrinsic matrix and distortion parameters
    CamCal = Calibration(cal_dir+"calibration.json", camera_names)

    #Get mask and ref file names
    MaskFiles = get_masks(cal_dir+"calibration.json", camera_names)

    #Load and undistort images
    for cname in camera_names:
        #load image
        img_file = img_dir + cname + "/" + cname + "_0037133.jpg"
        img = cv.imread(img_file)

        #undistort image
        dst = CamCal.undist(img, cname)
        cv.imwrite(os.getcwd()+"/"+cname+"_1.png", dst)

        #original image for comparison
        cv.imwrite(os.getcwd()+"/"+cname+"_0.png", img)

        #apply misalign reference
        #!!!SIZE MISMATCH!!!
        #mdir = cal_dir + "vehicle_database/VEHICLES/Jarvis/"
        #mis_ref = cv.imread(mdir + MaskFiles[cname]["mis_ref"])
        #mis_ref = np.array(mis_ref).astype(float)/np.max(mis_ref)
        #mr_dif = np.array(img)*mis_ref

        #cv.imwrite(os.getcwd()+"/"+cname+"mr.png", mr_dif.tolist())

