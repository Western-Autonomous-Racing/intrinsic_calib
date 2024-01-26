from typing import Any
import numpy as np
import cv_bridge 
import cv2

class Calibration:

    def __init__(self, bag, verify_calibration=True, time_offset=2, pattern_size=(7, 7), square_size=0.020):
        self.bag = bag
        self.bridge = cv_bridge.CvBridge()
        self.calib_images = []
        self.img_points = []
        self.objpoints = []
        self.test_image = None
        self.rectified_img = None
        self.img_width = 0
        self.img_height = 0
        self.verify_calibration = verify_calibration
        self.offset = time_offset
        self.pattern_size = pattern_size
        self.square_size = square_size

    def calibrate_images(self):
        
        self.get_pattern_corners(self.square_size, self.pattern_size)
        gray = cv2.cvtColor(self.calib_images[0], cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.img_points, gray.shape[::-1], None, None)
        
        if ret:
            print("Calibration successful.")
            if self.verify_calibration:
                self.validate_calibration(mtx, dist, rvecs, tvecs)

            # camera matrix, distortion coefficients, rotation and translation vectors
            return ret, mtx, dist, rvecs, tvecs, self.img_height, self.img_width
        else:
            print("Calibration failed.")
            return ret, None, None, None, None, 0, 0

    def validate_calibration(self, mtx, dist, rvecs, tvecs):
        h, w = self.test_image.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(self.test_image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        self.rectified_img = dst[y:y+h, x:x+w]
        cv2.imshow('Undistorted Image', self.rectified_img)
        cv2.waitKey(0)

        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("Total Reprojection Error: {}".format(mean_error/len(self.objpoints)))

    def get_pattern_corners(self, square_size, pattern_size, topic='/camera/image_raw', sample_size=300):
        # Iterate over the messages in the bag file
        
        print("Getting pattern corners...")

        num_frames = self.get_num_frames()
        init_image = False
        i=0
        floor = 0

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)*square_size

        duration = (self.bag.get_end_time() - self.bag.get_start_time()) - (self.offset * 2)
        between_frames = duration / (sample_size + 1)

        for topic, msg, t in self.bag.read_messages(topics=[topic]):

            # Convert the ROS image message to OpenCV format
            if (t.to_sec() - self.bag.get_start_time()) < self.offset:
                continue
            
            if i <= sample_size:

                print("Getting frame {} of {}".format((i + 1), sample_size))
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                if i == sample_size:
                    self.img_height, self.img_width = cv_image.shape[:2]
                    self.test_image = cv_image
                    break

                # Check if checkerboard pattern is detected
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('img', gray_image)
                # cv2.waitKey(1)

                found, corners = cv2.findChessboardCorners(gray_image, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
                print(found)
                if found:
                    # Do something with the image
                    print(f"Found pattern in frame {i} of {num_frames}")
                    corners2 = cv2.cornerSubPix(gray_image,corners, (11,11), (-1,-1), criteria)
                    
                    cv2.drawChessboardCorners(cv_image, pattern_size, corners2, found)
                    cv2.imshow('img', cv_image)
                    cv2.waitKey(500)

                    self.calib_images.append(cv_image)
                    self.img_points.append(corners2)
                    self.objpoints.append(objp)
            
                i+=1

        # Close the bag file
        self.bag.close()

    def get_num_frames(self, topic='/camera/image_raw'):
        num_frames = 0

        for topic, msg, t in self.bag.read_messages(topics=[topic]):
            num_frames += 1

        return num_frames
    
    def close(self):
        cv2.destroyAllWindows()
