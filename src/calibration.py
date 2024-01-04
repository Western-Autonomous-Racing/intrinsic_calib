from typing import Any
import numpy as np
import cv_bridge 
import cv2

class Calibration:

    def __init__(self, bag, verify_calibration=True):
        self.bag = bag
        self.bridge = cv_bridge()
        self.calib_images = []
        self.img_points = []
        self.objpoints = []
        self.test_image = None
        self.verify_calibration = verify_calibration

    def calibrate_images(self, pattern_size=(6, 6), square_size=0.108):
        
        self.get_pattern_corners(square_size, pattern_size=pattern_size)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.img_points, self.calib_images[0].shape[::-1], None, None)
        
        if self.verify_calibration:
            self.validate_calibration(mtx, dist, rvecs, tvecs)

    def validate_calibration(self, mtx, dist, rvecs, tvecs):
        h, w = self.test_image.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(self.test_image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('Undistorted Image', dst)
        cv2.waitKey(500)

        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("Total Error: {}".format(mean_error/len(self.objpoints)))

    def get_pattern_corners(self, square_size, topic='/camera/image_raw', sample_size=10, pattern_size=(6, 6)):
        # Iterate over the messages in the bag file
        
        num_frames = self.get_num_frames()
        init_image = False
        i=0
        j=0
        floor = 0
        frames_between = 0

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)*square_size

        for topic, msg, t in self.bag.read_messages(topics=[topic]):
            # Convert the ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if not init_image:

                # Check if checkerboard pattern is detected
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray_image, pattern_size)
                if found:
                    # Do something with the image
                    self.calib_images.append(cv_image)
                    self.img_points.append(corners)
                    self.objpoints.append(objp)
                    init_image = True
                    floor = i
                    j+=1
                    frames_between = int(num_frames / (sample_size+1))
            else:
                if j == sample_size:
                    self.test_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                elif i == floor + frames_between*j:
                    # Check if checkerboard pattern is detected
                    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCorners(gray_image, pattern_size)
                    if found:
                        # Do something with the image
                        self.calib_images.append(cv_image)
                        self.img_points.append(corners)
                        self.objpoints.append(objp)
                j+=1
            
            i+=1

        # Close the bag file
        self.bag.close()

    def get_num_frames(self, topic='/camera/image_raw'):
        num_frames = 0

        for topic, msg, t in self.bag.read_messages(topics=[topic]):
            num_frames += 1

        return num_frames
