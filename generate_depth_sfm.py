import os
import numpy as np
import cv2
import glob
import re

from calibration import Calibration

from tqdm import tqdm


def get_serial_list(npat):
    nums = [re.split('/|_|\.',s)[-2] for s in glob.glob(npat)]
    nums.sort()
    return nums


class DepthEstimator_sfm():
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


        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def load_image(self, camera_name, serial_number):
        #Read image file
        img_name = camera_name + "_" + serial_number + ".jpg"
        img = cv2.imread(
                os.path.join(self.img_dir, camera_name, img_name))
        #Undistort image
        img = self.CamCal.undistort(img, camera_name)
        return img

    @staticmethod
    def _form_transf(R, t):
        """
        Creates a transformation matrix from the given rotation matrix and 
        translation vector

        Parameters
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, img_c, img_p):
        """
        This function detects and computes keypoints and descriptors from 
        the previous and current frames using the class orb object

        Parameters
        img_c: current frame
        img_p: previous frame

        Returns
        q1 (ndarray): good keypoint matches in previous frame
        q2 (ndarray): good keypoint matches in current frame
        """
        #Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img_p, None)
        kp2, des2 = self.orb.detectAndCompute(img_c, None)
        #Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        #Find the matches that do not have a large distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, #draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, #draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(img_c, kp1, img_p, 
                               kp2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(0)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2, cname):
        """
        Calculates the transformation matrix

        Parameters
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, 
                    self.CamCal.Intrinsic_UD[cname], threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2, cname)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2, cname):
        """
        Decompose the Essential matrix

        Parameters
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            #Get the transformation matrix
            T = self._form_transf(R, t)
            #Make the projection matrix
            CamMat = self.CamCal.Intrinsic_UD[cname]
            P = np.concatenate((CamMat, np.zeros((3, 1))), axis=1) @ T
            #T = np.hstack([R, t.reshape(3,1)])
            #P = CamMat @ T

            #Triangulate the 3D points
            ProMat = np.hstack([CamMat,np.zeros((3,1))])
            hom_Q1 = cv2.triangulatePoints(ProMat, P, q1.T, q2.T)
            #Also seen from cam 2
            hom_Q2 = T @ hom_Q1

            #Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            #Find the number of points there has positive z coordinate 
            #in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        #Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        #Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        #Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        #Select the pair that has the most points with positive 
        #z coordinates
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def main():
    dat_dir = os.path.join(os.getcwd(), "datafiles/20220422-133712-00.40.45-00.41.45@Jarvis/sensor")

    main_cam_name = "F_MIDLONGRANGECAM_CL"
    other_cam_names = ["B_MIDRANGECAM_C",
                       "F_MIDRANGECAM_C",
                       "M_FISHEYE_L",
                       "M_FISHEYE_R"]
    other_cam_names = ["B_MIDRANGECAM_C", "F_MIDRANGECAM_C"]

    #Set up generator
    gen = DepthEstimator_sfm(dat_dir, main_cam_name, other_cam_names)

    count = 0
    for cname in [main_cam_name]:
        out_dir = os.path.join(gen.dep_dir, cname+"_vo")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        cur_pose = np.eye(4)
        input_img = None
        previous = None
        for snum in gen.serials:
            count+=1
            if count < 840:
                if count == 839:
                    previous = gen.load_image(cname,snum)
                continue
            elif count > 850:
                break
            elif count == 840:
                print(count, snum)
            else:
                previous = input_img
                print(count, snum)

            input_img = gen.load_image(cname, snum)
            q1, q2 = gen.get_matches(input_img, previous)
            transf = gen.get_pose(q1, q2, cname)
            cur_pose = cur_pose @ np.linalg.inv(transf)
            print(f"Current pose: \n{cur_pose}\n")


        print(f"Depth maps for {cname} ready")

if __name__ == "__main__":
    main()
