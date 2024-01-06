import argparse
import rosbag
from calibration import Calibration
import yaml
import os
import shutil


def get_bag(bag_path):
    bag = rosbag.Bag(bag_path)
    return bag

def write_cals(mtx, dist, img_height, img_width, path):
    with open(path, 'r') as f:
        cals = yaml.load(f, Loader=yaml.FullLoader)

                        #fu     fv      u0      v0
    cals["intrinsics"] = [mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]]
    cals["distortion_coeffs"] = [dist[0], dist[1], dist[0][2], dist[0][3], dist[0][4]]
    cals["resolution"] = [img_width, img_height]

    with open(path, "w") as f:
        yaml.dump(cals, f)

def create_new_cal(cal_name):
    template_path = ".../template/chain_template.yaml"
    output_path = f".../output_cals/{cal_name}.yaml"
    shutil.copyfile(template_path, output_path)
    return output_path

def main():
    # Parse command line arguments

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--bag', help='Path to ROS bag file')
        parser.add_argument('-n', '--name', help='Name of calibration file (default: chain.yaml)', default='chain.yaml')

        args = parser.parse_args()

        # Open the ROS bag file
        bag = rosbag.Bag(args.bag)
        
        print("Opened bag file")

        calibration = Calibration(bag)
        ret, mtx, dist, rvecs, tvecs, img_height, img_width = calibration.calibrate_images()

        if ret:
            if not os.path.exists(".../output_cals"):
                os.mkdir("../output_cals")
            
            output_cal = create_new_cal(args.name)

            write_cals(mtx, dist, img_height, img_width, output_cal)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down")
        exit(0)


if __name__ == "__main__":
    main()
