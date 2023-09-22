#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import message_filters
import open3d as o3d
import cv2
# import matplotlib.pyplot as plt
# from CMap2D import CMap2D, gridshow
# import imageio as Image
# import glob
# from scipy.spatial.transform import Rotation as R
import os
# from scipy.special import softmax
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
# from PIL import Image as image
from cv_bridge import CvBridge
from esanet import get_gradients
import tf2_ros

def homogenous(trans, rotmat):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotmat
    homogeneous_matrix[:3, 3:] = trans

    return homogeneous_matrix

def image_callback(rgb_msg, depth_msg, camera_msg):

    bridge = CvBridge()
    
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
    # Convert depth image from ROS message to numpy array
    depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape((depth_msg.height, depth_msg.width))
    

    h = camera_msg.height
    w = camera_msg.width
    name1 = camera_msg.header.stamp.secs
    time = camera_msg.header.stamp
 
    intrinsic = np.array(camera_msg.K).reshape(3,3)
    distortion = np.array(camera_msg.D)
    undistorted_rgb_image = cv2.undistort(rgb_image, intrinsic, distortion)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics( width=w, height=h, fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2])



    # grad = get_gradients(rgb_image, depth_image, h, w)
    
    # create point clouds

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(undistorted_rgb_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)

    # Create Open3D point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
    
    
    
    print(time)
    trans2 = tfBuffer.lookup_transform_full( 'map', time, 'camera_link', time, 'map', rospy.Duration(10.0))
    while not rospy.is_shutdown():

            try:

                trans1 = tfBuffer.lookup_transform('camera_link', 'camera_depth_optical_frame', rospy.Time(0))
                trans2 = tfBuffer.lookup_transform_full( 'map', camera_msg.header.stamp, 'camera_link', camera_msg.header.stamp, 'map', rospy.Duration(1.0))

                name2 = trans2.header.stamp.secs
                transl = np.array([[trans1.transform.translation.x], [trans1.transform.translation.y], [trans1.transform.translation.z]])
                rotmat = pcd.get_rotation_matrix_from_quaternion([trans1.transform.rotation.x, trans1.transform.rotation.y, trans1.transform.rotation.z, trans1.transform.rotation.w])
                transform_mat = homogenous(transl, rotmat)
                pcd = pcd.transform(transform_mat)

                transl = np.array([[trans2.transform.translation.x], [trans2.transform.translation.y], [trans2.transform.translation.z]])
                rotmat = pcd.get_rotation_matrix_from_quaternion([trans2.transform.rotation.x, trans2.transform.rotation.y, trans2.transform.rotation.z,trans2.transform.rotation.w])
                transform_mat = homogenous(transl, rotmat)
                pcd = pcd.transform(transform_mat)
                
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
    o3d.io.write_point_cloud(f"/home/ajay/work/msc_project/git/clouds/{name1}_{name2}.ply", pcd)
    
# Initialize the ROS node
rospy.init_node('seg_map')

# tf_listener = tf.TransformListener()
rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
camera_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)

tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
listener = tf2_ros.TransformListener(tfBuffer)
# map_sub = message_filters.Subscriber('/rtbmap/proj_map', OccupancyGrid)
# while not rospy.is_shutdown():
#     try:
#         (rgb_trans,rgb_rot) = tf_listener.lookupTransform('camera_rgb_frame', 'map', rospy.Time(0))
#         (depth_trans, depth_rot) = tf_listener.lookupTransform('camera_depth_frame', 'map', rospy.Time(0))
#     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#         continue


ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_sub], 30, 0.1)
ts.registerCallback(image_callback)
# rospy.on_shutdown(shutdown)
# Spin ROS
rospy.spin()

