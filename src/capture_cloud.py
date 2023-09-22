#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import open3d as o3d
import message_filters
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped 
import matplotlib.pyplot as plt
import tf
import copy

first_frame_received = False

def image_callback(rgb_msg, depth_msg, imu_msg):
    global name
    global first_frame_received 
    listener = tf.TransformListener()

    # print('callback')
    T = np.eye(4)
    fx, fy, cx, cy = 911.8645629882812, 910.4451904296875, 644.7821655273438, 366.3886413574219
    # Convert RGB image from ROS message to numpy array
    bridge = CvBridge()
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
    name_secs = rgb_msg.header.stamp.secs
    name_nsecs = rgb_msg.header.stamp.nsecs

    # Convert depth image from ROS message to numpy array
    depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))

    # Create Open3D RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)

    # Create Open3D point cloud from RGBD image
    intrinsic = o3d.camera.PinholeCameraIntrinsic(rgb_image.shape[1], rgb_image.shape[0], fx, fy, cx, cy)

    if not first_frame_received:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        #transform from camera_color_frame to camera_link 
        while not rospy.is_shutdown():
            try:
                # trans, rot = listener.lookupTransform('camera_color_frame', 'camera_link',  rospy.Time(0))
                # rotmat = pcd.get_rotation_matrix_from_quaternion(rot)
                # pcd.rotate(rotmat, center=(0,0,0))
                # pcd.translate(trans)
                # o3d.visualization.draw_geometries([pcd])
                first_frame_received = True
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
    else:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        #transform from camera_color_frame to camera_link
        while not rospy.is_shutdown():
            try: 
                # trans1, rot1 = listener.lookupTransform('camera_color_frame', 'camera_link',  rospy.Time(0))
                # rotmat1 = pcd.get_rotation_matrix_from_quaternion(rot1)
                # pcd.rotate(rotmat1, center=(0,0,0))
                # pcd.translate(trans1)


                # trans2 = imu_msg.pose.position
                # rot2 = imu_msg.pose.orientation
                # rotmat2 = pcd.get_rotation_matrix_from_quaternion((rot2.x, rot2.y, rot2.z, rot2.w))
                # pcd.rotate(rotmat2, center=(0,0,0))
                # pcd.translate((trans2.x, trans2.y, trans2.z))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue



    o3d.io.write_point_cloud(name+"{}_{}.ply".format(name_secs, name_nsecs), pcd)
    


# Initialize the ROS node
rospy.init_node('open3d_ros')
name = '/home/ajay/work/dmar/data/rgbd1/'

rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
imu_pose_sub = message_filters.Subscriber('/Imu_pose', PoseStamped)

# print('started')
ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, imu_pose_sub], 100, 1)
ts.registerCallback(image_callback)
# rospy.on_shutdown(shutdown)
# Spin ROS
rospy.spin()