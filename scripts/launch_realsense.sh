roslaunch open3d_ros rs_rgbd.launch unite_imu_method:=copy \
          enable_gyro:=true enable_accel:=true imu_processing:=true \
          enable_sync:=true hold_back_imu_for_frames:=true \
          publish_tf:=true tf_publish_rate:=30 publish_odom_tf:=true 

# rosrun Imu_Integrator Imu_Integrator_node &

# rosrun rviz rviz 