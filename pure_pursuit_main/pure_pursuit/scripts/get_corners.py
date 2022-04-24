#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

# TODO CHECK: include needed ROS msg type headers and libraries

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import message_filters

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

        self.waypoints       =       np.load("/sim_ws/src/pure_pursuit/scripts/waypoints2.npy")
        # self.waypoints      =       self.waypoints[:, 0:1000:20]
        # print("Type self.waypoints: ", type(self.waypoints))
        # print("Shape of self.waypoints: " , self.waypoints.shape)
        # print("Type of self.waypoints[0]: ", type(self.waypoints[0]))
        # print("Length of self.waypoints[0]: ", len(self.waypoints[0]))

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

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.visualize_pub.publish(self.vis_msg)

        odomTopic = "/ego_racecar/odom"
        scan_topic = "/scan"

        self.drivePub = self.create_publisher(AckermannDriveStamped,"drive",0)
        # self.odomSub = self.create_subscription(Odometry,odomTopic,self.pose_callback,0)

        self.scan_sub     = message_filters.Subscriber(scan_topic,LaserScan)#,self.raw_callback)
        self.odom_sub     = message_filters.Subscriber(odomTopic,Odometry)#,self.depth_callback)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.odoom_sub, self.scan_sub], 300, 1.0)
        self.ts.registerCallback(self.callback)
        
        self.ld = 0.7 #lookahead distance constant to 0.5m



    # def timer_callback(self):
    #     self.visualize_pub.publish(self.vis_msg)


    def callback(self, pose_msg,lidar_msg):
        # pass
        # TODO: find the current waypoint to track using methods mentioned in lecture
        # currPosex = pose_msg.twist.linear.x
        # currPosey = pose_msg.twist.linear.y #Gets the x and y values of my current pose
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


        bool_in_circle = distArray <= self.ld
        bool_out_circle = distArray >= self.ld

        gPts_out = gPts[:,bool_out_circle]
        out_pt = gPts_out[:, np.argmin(distArray[bool_out_circle])]

        if(bool_in_circle.sum() != 0):
            gPts_in = gPts[:,bool_in_circle]
            in_pt = gPts_in[:, np.argmax(distArray[bool_in_circle])]
        else:
            in_pt = out_pt

        '''
        Gets wr,wl for our purposes
        '''


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
        goalPt = (out_pt + in_pt)/2
        x = goalPt[0]
        y = goalPt[1]
        
        # TODO: calculate curvature/steering angle
        curvature = 2*goalPt[1]/(self.ld**2)
        # TODO: publish drive message, don't forget to limit the steering angle.
        msg = AckermannDriveStamped()
        msg.drive.speed = float(2.0)
        msg.drive.steering_angle = curvature
        # if(msg.drive.steering_angle>np.radians(20))

        self.drivePub.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
