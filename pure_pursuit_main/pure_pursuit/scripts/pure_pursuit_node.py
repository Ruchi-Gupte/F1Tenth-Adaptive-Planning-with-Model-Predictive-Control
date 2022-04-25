#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

# TODO CHECK: include needed ROS msg type headers and libraries

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseStamped
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
import time
from scipy.spatial.transform import Rotation as R
import copy

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')



        """
        self.waypoints: Load the waypoint saved 

            Type: numpy array -> Shape : [2,1000] where 2 corresponds to x and y and 1000 are the number of points
        """
        self.on_car =   False
        
        '''
        1:      oldest non_optimized
        2:      clipped at 0.5 both sides
        3:      clipped at 0.7 both sides
        4:      clipped-> left->0.3   ,   right->0.7
        5:      clipped-> left->0.7   ,   right->0.9
        '''

        trajectory          =       6
        if(self.on_car):
            if(trajectory == 1):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory22April.npy"
            elif(trajectory == 2):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.5.npy"
            elif(trajectory == 3):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.7.npy"
            elif(trajectory == 4):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.3_0.7.npy"
            elif(trajectory == 5):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.7_0.9.npy"
            elif(trajectory == 6):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.3_0.95.npy"
            elif(trajectory == 7):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.5_0.95.npy"
            elif(trajectory == 8):
                traj_path       =       "/f1tenth_ws/src/pure_pursuit/scripts/trajectory_0.0_0.95.npy"
        else:
            if(trajectory == 1):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory22April.npy"
            elif(trajectory == 2):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.5.npy"
            elif(trajectory == 3):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.7.npy"
            elif(trajectory == 4):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.3_0.7.npy"
            elif(trajectory == 5):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.7_0.9.npy"
            elif(trajectory == 6):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.3_0.95.npy"
            elif(trajectory == 7):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.5_0.95.npy"
            elif(trajectory == 8):
                traj_path       =       "/sim_ws/src/pure_pursuit/scripts/trajectory_0.0_0.95.npy"



        if(self.on_car):
            odomTopic = "/pf/viz/inferred_pose"
            self.drivePub = self.create_publisher(AckermannDriveStamped,"drive",0)
            self.odomSub = self.create_subscription(PoseStamped,odomTopic,self.pose_callback,0)
        else:
            odomTopic = "/ego_racecar/odom"
            self.drivePub = self.create_publisher(AckermannDriveStamped,"drive",0)
            self.odomSub = self.create_subscription(Odometry,odomTopic,self.pose_callback,0)

        self.points          =       np.load(traj_path)


        self.waypoints      =       self.points.T[:2]           #Shape: (2,N)
        self.speed          =       self.points.T[2]            #Shape: (N,)
        self.curvature      =       np.abs(self.points.T[4])    #Shape: (N,)
        
        # TODO: create ROS subscribers and publishers

        vis_topic = "visualization_marker"

        self.visualize_pub              =       self.create_publisher(Marker, vis_topic, 10)

        self.vis_msg                         =       Marker()
        self.vis_msg.header.frame_id         =       "/map"
        self.vis_msg.type                    =       Marker.POINTS
        self.vis_msg.action                  =       Marker.ADD
        self.vis_msg.scale.x                 =       0.1
        self.vis_msg.scale.y                 =       0.1
        self.vis_msg.scale.z                 =       0.1
        self.vis_msg.color.g                 =       1.0
        self.vis_msg.color.a                 =       1.0
        # self.vis_msg.pose.position.x         =       self.waypoints[0,0]
        # self.vis_msg.pose.position.y         =       self.waypoints[1,0]
        # self.vis_msg.pose.position.z         =       0.0
        # self.vis_msg.pose.position.x         =       1.92
        # self.vis_msg.pose.position.y         =       0.0448
        # self.vis_msg.pose.position.z         =       1.0
        self.vis_msg.pose.orientation.w      =       1.0
        self.vis_msg.lifetime                =       Duration()
        # self.vis_msg.lifetime.sec            =       0               

        for i in range(self.waypoints.shape[1]):
        # for i in range(1):
            p       =       Point()

            p.x     =       self.waypoints[0,i]
            p.y     =       self.waypoints[1,i]
            p.z     =       0.0

            self.vis_msg.points.append(p)

        # timer_period = 0.5
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.visualize_pub.publish(self.vis_msg)

        ld_point_topic                  = "visualization_marker2"
        self.ld_point_vis               =       self.create_publisher(Marker, ld_point_topic, 10)
        self.ld_point_msg               =       copy.deepcopy(self.vis_msg)
        self.ld_point_msg.points        =       []
        self.ld_point_msg.color.g       =       0.0
        self.ld_point_msg.color.r       =       1.0
        self.ld_point_msg.scale.x       =       0.2
        self.ld_point_msg.scale.y       =       0.2
        self.ld_point_msg.scale.z       =       0.2

        
        min_curvature       =       self.curvature.min()
        max_curvature       =       self.curvature.max()
        max_ld              =       2.3
        min_ld              =       1.5

        self.ld             =       (min_ld - max_ld)/(max_curvature - min_curvature) #lookahead distance constant to 0.5m
        self.ld             =       self.ld*(self.curvature - min_curvature)
        self.ld             =       self.ld + max_ld

        self.curr_ld        =       max_ld
        self.prev_time      =       time.time()


    # def timer_callback(self):
    #     self.visualize_pub.publish(self.vis_msg)


    def pose_callback(self, pose_msg):
        # pass
        # TODO: find the current waypoint to track using methods mentioned in lecture
        # currPosex = pose_msg.twist.linear.x
        # currPosey = pose_msg.twist.linear.y #Gets the x and y values of my current pose

        self.visualize_pub.publish(self.vis_msg)

        if self.on_car:
            currPosex = pose_msg.pose.position.x
            currPosey = pose_msg.pose.position.y
            currPose = np.array([currPosex,currPosey,0]).reshape((3,-1))
            qx = pose_msg.pose.orientation.x
            qy = pose_msg.pose.orientation.y
            qz = pose_msg.pose.orientation.z
            qw = pose_msg.pose.orientation.w
        else:
            time.sleep(0.018)
            currPosex = pose_msg.pose.pose.position.x
            currPosey = pose_msg.pose.pose.position.y
            currPose = np.array([currPosex,currPosey,0]).reshape((3,-1))
            qx = pose_msg.pose.pose.orientation.x
            qy = pose_msg.pose.pose.orientation.y
            qz = pose_msg.pose.pose.orientation.z
            qw = pose_msg.pose.pose.orientation.w
        
        rot_car_world = R.from_quat([qx,qy,qz,qw])
        
        roll,pitch,yaw = rot_car_world.as_euler('xyz',degrees=False)
        # print("Roll = :", roll)
        # print("pitch = :", pitch)
        # print("yaw = :", yaw)
        
        wPts = self.waypoints
        wPts = np.vstack((wPts,np.zeros((1,wPts.shape[1]))))
        gPts = rot_car_world.apply((wPts - currPose).T,inverse=True) 
        gPts = gPts[:,:2].T #This is expected to be of shape (2xN) -> sanity check. If not, check the conversion wrt robot frame CHECKPT
        

        gPts = gPts[:, gPts[0,:]>0 ]
        distArray = np.linalg.norm(gPts,axis = 0) 


        bool_in_circle = distArray <= self.curr_ld
        bool_out_circle = distArray >= self.curr_ld

        gPts_out = gPts[:,bool_out_circle]
        out_pt = gPts_out[:, np.argmin(distArray[bool_out_circle])]

        if(bool_in_circle.sum() != 0):
            gPts_in = gPts[:,bool_in_circle]
            in_pt = gPts_in[:, np.argmax(distArray[bool_in_circle])]
        else:
            in_pt = out_pt

        # closest_out_of_circle_point = np.argmin(distArray[distArray >= self.ld])

        # bool_in_circle = distArray <= self.ld
        # # print(distArray.shape)
        # # print(bool_in_circle.sum())
        # if(bool_in_circle.sum() != 0):
        #     farthest_in_circle_point    = np.argmax(distArray[bool_in_circle])

        # else:
        #     farthest_in_circle_point = closest_out_of_circle_point

        # out_pt = gPts[:,closest_out_of_circle_point]
        # in_pt = gPts[:,farthest_in_circle_point]
        self.ld_point_msg.points        =       []
        goalPt = (out_pt + in_pt)/2
        x = goalPt[0]
        y = goalPt[1]
        
        # import pdb;pdb.set_trace()
        p_world = rot_car_world.apply(np.array([x,y,0.0])).reshape(-1,1) + currPose
        p       =       Point()
        p.x     =       p_world[0][0]
        p.y     =       p_world[1][0]
        p.z     =       0.0
        self.ld_point_msg.points.append(p)
        self.ld_point_vis.publish(self.ld_point_msg)

        # TODO: calculate curvature/steering angle
        curvature = 2*goalPt[1]/(self.curr_ld**2)
        # TODO: publish drive message, don't forget to limit the steering angle.

        speedIdx = np.argmin(np.linalg.norm(self.waypoints - np.array([currPosex,currPosey]).reshape((-1,1)),axis = 0))
        traj_curvature = self.curvature[speedIdx]

        # print("Trajectory Curvature: ",traj_curvature, "  ld: ", self.curr_ld)

        self.curr_ld   = self.ld[speedIdx]

        msg = AckermannDriveStamped()
        msg.drive.speed = float(self.speed[speedIdx])
        # msg.drive.speed =   1.0
        msg.drive.steering_angle = curvature
        # if(msg.drive.steering_angle>np.radians(20))

        self.drivePub.publish(msg)
        print("Speed: ", msg.drive.speed)
        # print("Publishing Frequency: ", 1/(time.time() - self.prev_time))
        self.prev_time = time.time()
        
def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
