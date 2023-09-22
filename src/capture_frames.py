#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Initialize the ROS node
class capture_frames():
    def __init__(self):
        
        self.name = '/home/ajay/work/dmar/calibration/frames/'
        self.image_saved = False
        rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
  
    def image_callback(self,frames):
        if not self.image_saved:
            bridge = CvBridge()
            rgb_image = bridge.imgmsg_to_cv2(frames, desired_encoding='rgb8')
            name_secs = frames.header.stamp.secs
            name_nsecs = frames.header.stamp.nsecs
            cv2.imwrite(self.name+f'{name_secs}_{name_nsecs}.jpg', rgb_image[:,:,::-1])
            rospy.signal_shutdown("Image saved, shutting down node")  


def main():
    rospy.init_node('capture_frames')
    capture_frames()
    rospy.spin()

if __name__ == "__main__":
    main()