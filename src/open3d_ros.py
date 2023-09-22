#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
import open3d as o3d
import message_filters
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose 
import matplotlib.pyplot as plt

# initial values
C_minus = np.eye(3)
time_prev = 0.0
g0 = np.zeros((3,1))
first_frame_received = False
vi_minus = np.array([[0.0],
                    [0.0],
                    [0.0]])

ri_minus = np.array([[0.0],
                     [0.0],
                     [0.0]])

orientation = []
position = []
velocity = []

def skew_mat(wx, wy, wz, dt):
    skew = np.array([[1, -wz*dt, wy*dt],
                     [wz*dt, 1, -wx*dt],
                     [-wy*dt, wx*dt, 1]])
    return skew

def attitude_update(C_, omgx, omgy, omgz, dt):
    Iomega_dt = np.array([[1, -omgz * dt, omgy * dt],
                     [omgz * dt, 1, -omgx * dt],
                     [-omgy * dt, omgx * dt, 1]])
    
    C = C_ @ Iomega_dt
    return C

def specific_force_transform(C_, C, accx, accy, accz):
    f = np.array([[accx],
         [accy],
         [accz]])
    f = f.reshape((3,1))
    fib = ((1/2) * (C_ + C)) @ f
    return fib

def velocity_update(V_, fib, g0, dt):
    aib = fib + g0
    V = V_ + (aib * dt)
    return V, aib

def position_update(r_, V, aib, dt):
    r = r_ + (V * dt) - (aib * ((dt**2)/2))
    return r 

def shutdown():
    global position
    position = np.array(position)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')



    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].


    ax.scatter(position[:,0], position[:,1], position[:,2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def dead_reckoning(imu):
    global C_minus
    global time_prev
    global first_frame_received
    global vi_minus
    global ri_minus
    global g0
    global orientation
    global position
    global velocity

    # print(imu)
    # consider first frame as stationary
    if not first_frame_received:
        first_frame_received=True
        g0 = np.array([[imu.angular_velocity.x],
                       [imu.angular_velocity.y],
                       [imu.angular_velocity.z]]) 
        
        time_prev = imu.header.stamp.secs * 10**9 + imu.header.stamp.nsecs
        rotmat = None
        translation = None

    else:
        ang_vel_x = imu.angular_velocity.x
        ang_vel_y = imu.angular_velocity.y
        ang_vel_z = imu.angular_velocity.z
        lin_acc_x = imu.linear_acceleration.x
        lin_acc_y = imu.linear_acceleration.y
        lin_acc_z = imu.linear_acceleration.z
        time_curr = imu.header.stamp.secs * 10**9 + imu.header.stamp.nsecs
        dt        = (time_curr - time_prev) / 10**9

        # attitude update
        

        C_plus = attitude_update(C_minus, ang_vel_x, ang_vel_y, ang_vel_z, dt)
        F = specific_force_transform(C_minus, C_plus, lin_acc_x, lin_acc_y, lin_acc_z)
        vi_plus, aib = velocity_update(vi_minus, F, g0, dt)
        ri_plus = position_update(ri_minus, vi_plus, aib, dt)
        
        
        orientation.append(C_plus)
        position.append(ri_plus)
        velocity.append(vi_plus)
        rotmat = np.linalg.inv(C_minus) @ C_plus
        translation = ri_plus - ri_minus
        # print(f"rotmat_T: \n{rotmat.T}")
        # print(f"rotmat_inv: \n {np.linalg.inv(rotmat)}")
        ri_minus = ri_plus
        vi_minus = vi_plus
        C_minus = C_plus
        time_prev = time_curr

        return rotmat, translation
          
def image_callback(rgb_msg, depth_msg, imu_msg):
    global name

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
        # print(imu_msg)
        dead_reckoning(imu_msg)
    else:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        rotmat, translation = dead_reckoning(imu_msg)
        # rotmat = R.from_matrix(rotmat)
        # rotmat_euler = rotmat.as_euler('xyz', degrees=False)
        # rotmat = pcd.get_rotation_matrix_from_xyz(rotmat_euler)
        T[:3,:3] = rotmat
        T[:3, 3] = translation[:,0]
        points = np.asarray(pcd.points)
        ones = np.ones((points.shape[0], 1))
        points_homo = np.column_stack((points, ones))
        points_transform = (T @ points_homo.T).T
        pcd.points = o3d.utility.Vector3dVector(points_transform[:,:3])
        
        print(points.shape)

    # o3d.io.write_point_cloud(name+"{}_{}.ply".format(name_secs, name_nsecs), pcd)
    


# Initialize the ROS node
rospy.init_node('open3d_ros')

name = '/home/ajay/work/dmar/data/rgbd/'

rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
imu_sub = message_filters.Subscriber('/camera/imu', Imu)


ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, imu_sub], 100, 0.001)
ts.registerCallback(image_callback)
rospy.on_shutdown(shutdown)
# Spin ROS
rospy.spin()
