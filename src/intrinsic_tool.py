import argparse
import rosbag
from calibration import Calibration


def get_bag(bag_path):
    bag = rosbag.Bag(bag_path)
    return bag

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', help='Path to ROS bag file')

    args = parser.parse_args()

    # Open the ROS bag file
    bag = rosbag.Bag(args.bag)

    calibration = Calibration(bag)
    calibration.calibrate_images()

    
if __name__ == "__main__":
    main()
