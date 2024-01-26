import argparse
import rosbag
from calibration import Calibration
import yaml
import os
import shutil
from datetime import datetime
import cv2


def get_bag(bag_path):
    bag = rosbag.Bag(bag_path)
    return bag

# Define a custom representer for lists
def list_representer(dumper, data):
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

def write_cals(mtx, dist, img_height, img_width, path):

    # Register the custom representer
    yaml.add_representer(list, list_representer)    


    with open(path, 'r') as f:
        cals = yaml.safe_load(f)

    # Convert the values to regular floats
    cals["cam0"]["intrinsics"] = [float(mtx[0][0]), float(mtx[1][1]), float(mtx[0][2]), float(mtx[1][2])]
    cals["cam0"]["distortion_coeffs"] = [float(dist[0][0]), float(dist[0][1]), float(dist[0][2]), float(dist[0][3])]
    cals["cam0"]["resolution"] = [int(img_width), int(img_height)]

    with open(path, "w") as f:
        yaml.dump(cals, f, default_flow_style=False, width=float("inf"))

def save_imgs(path, test_img, rectified_img):
    print("Saving images...")
    cv2.imwrite(f"{path}/test_img.png", test_img)
    cv2.imwrite(f"{path}/rectified_img.png", rectified_img)

def create_new_cal(cal_name):
    current_datetime = datetime.now()
    cal_folder = current_datetime.strftime("%Y-%m-%d-%TH-%M-%S")
    
    os.mkdir(f"../output_cals/{cal_folder}")
    template_path = "../template/chain_template.yaml"
    output_path = f"../output_cals/{cal_folder}/{cal_name}.yaml"
    shutil.copyfile(template_path, output_path)
    return output_path, cal_folder
    

def main():
    # Parse command line arguments

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--bag', help='Path to ROS bag file')
        parser.add_argument('-s', '--save', help='Save calibration file to output_cals directory', action='store_true')
        parser.add_argument('-n', '--name', help='Name of calibration file (default: chain.yaml)', default='chain')

        args = parser.parse_args()

        # Open the ROS bag file
        bag = rosbag.Bag(args.bag)
        
        print("Opened bag file")

        calibration = Calibration(bag, args.save)
        ret, mtx, dist, rvecs, tvecs, img_height, img_width = calibration.calibrate_images()

        if ret:
            if not os.path.exists("../output_cals"):
                os.mkdir("../output_cals")
            
            output_cal, cal_folder = create_new_cal(args.name)

            write_cals(mtx, dist, img_height, img_width, output_cal)

            if args.save:
                save_imgs(f"../output_cals/{cal_folder}", calibration.test_image, calibration.rectified_img)

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down")
        exit(0)


if __name__ == "__main__":
    main()
