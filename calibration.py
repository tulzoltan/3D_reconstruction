import json
import numpy as np
import cv2
import os


class Calibration():
    def __init__(self, 
                 directory, 
                 file_name,
                 camera_names):
        """Get intrinsic matrix and distortion parameters from calibration file"""

        calibration_file = os.path.join(directory, file_name)
        vehicle_dir = "vehicle_database/VEHICLES/Jarvis"

        self.width = {}
        self.height = {}
        self.Intrinsic = {}
        self.Intrinsic_UD = {}
        self.Adjust_UD = {}
        self.Distortion = {}
        self.Extrinsic = {}
        self.ObstructionMask = {}
        self.bottom_crop = {}

        with open(calibration_file) as in_file:
            data = json.load(in_file)
            for cname in camera_names:
                cdata = data[cname]

                #Image resolution
                self.width[cname], self.height[cname] = cdata["image_resolution_px"]

                #Intrinsic matrix
                self.Intrinsic[cname] = np.eye(3)
                self.Intrinsic[cname][0,0], self.Intrinsic[cname][1,1] = cdata["focal_length_px"]
                self.Intrinsic[cname][0,2], self.Intrinsic[cname][1,2] = cdata["principal_point_px"]

                #Distortion parameters
                self.Distortion[cname] = np.array(cdata["distortion_coeffs"])

                #Intrinsic matrix after rectification
                #Not good for FISHEYE
                newCamMat, roi = cv2.getOptimalNewCameraMatrix(
                            self.Intrinsic[cname], 
                            self.Distortion[cname], 
                            (self.width[cname], self.height[cname]), 
                            1, 
                            (self.width[cname], self.height[cname]))
                self.Intrinsic_UD[cname] = newCamMat
                self.Adjust_UD[cname] = roi
                if not "FISHEYE" in cname:
                    _, _, self.width[cname], self.height[cname] = roi

                #Extrinsic matrix
                self.Extrinsic[cname] = np.array(cdata["RT_sensor_from_body"])
                #self.Extrinsic[cname] = np.array(cdata["RT_body_from_sensor"])

                #Obstruction mask
                fname = cdata["custom_vars"]["obstruction_mask_file"]
                obs_mask = cv2.imread(
                        os.path.join(directory, vehicle_dir, fname))
                rule = obs_mask < 255/2
                obs_mask[rule] = 0
                obs_mask[~rule] = 1
                obs_mask = self.undistort(obs_mask, cname)
                self.ObstructionMask[cname] = obs_mask
                self.bottom_crop[cname] = int(np.argmax(np.any(obs_mask[::-1], axis=(1, 2))))
                if "FISHEYE" not in cname:
                    self.height[cname] -= self.bottom_crop[cname]

    def undistort(self, img_in, cam_name):
        """Undistort and crop image, handle fisheye cameras separately"""
        h, w = img_in.shape[:2]
        CamMat = self.Intrinsic[cam_name]
        newCamMat = self.Intrinsic_UD[cam_name]

        if "FISHEYE" in cam_name:
            Dist = self.Distortion[cam_name][:-1]
            #newCamMat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(CamMat, Dist, (w,h), np.eye(3))
            #Undistort
            mapx, mapy = cv2.fisheye.initUndistortRectifyMap(CamMat, Dist, None, CamMat, (w,h), cv2.CV_16SC2) #or cv2.CV_32FC1
            img_out = cv2.remap(img_in, mapx, mapy,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

        else:
            Dist = self.Distortion[cam_name]

            #Undistort
            #img_out = cv2.undistort(img_in, CamMat, Dist, None, newCamMat)

            #Undistort using remap
            mapx, mapy = cv2.initUndistortRectifyMap(
                        CamMat, Dist, None, newCamMat, (w,h), 5)
            img_out = cv2.remap(img_in, mapx, mapy,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

            # crop image
            #x, y, w, h = self.Adjust_UD[cam_name]
            #img_out = img_out[y:y+h-self.bottom_crop[cam_name], x:x+w]

        return img_out


    def crop_image(self, img_in, cam_name):
        if "FISHEYE" not in cam_name:
            x, y, w, h = self.Adjust_UD[cam_name]
            img_out = img_in[y:y+h, x:x+w]

            #use obstruction masks
            bottom_crop = img_out.shape[0] - self.bottom_crop[cam_name]
            img_out = img_out[:bottom_crop]

            return img_out

        else:
            return img_in



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
    CamCal = Calibration(cal_dir, "calibration.json", camera_names)

    #Load and undistort images
    for cname in camera_names:
        print(cname, CamCal.width[cname], CamCal.height[cname])

        mask = CamCal.ObstructionMask[cname]
        x2 = np.argmax(np.all(mask[::-1], axis=(1,2)))

        frames = ["0037448"]
        #frames = ["0037448", "0037133"]
        for frame in frames:
            #load image
            img_file = img_dir + cname + "/" + cname + "_" +frame+".jpg"
            img = cv2.imread(img_file)

            #undistort image
            dst = CamCal.undistort(img, cname)
            cv2.imwrite(os.getcwd()+"/"+cname+"_"+frame+"_1.png", dst)
            dst = CamCal.crop_image(dst, cname)
            cv2.imwrite(os.getcwd()+"/"+cname+"_"+frame+"_2.png", dst)

            #original image for comparison
            cv2.imwrite(os.getcwd()+"/"+cname+"_"+frame+"_0.png", img)

