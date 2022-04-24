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
from scipy.spatial.transform import Rotation as R

import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev

pts = np.array([
[9.15,0.554],
[9.28,1.89],
[9.27,4.19],
[9.19,7.65],
[9.37,9.31],
[8.98,11.1],
[7.65,11.6],
[6.37,11.2],
[6.11,10.5],
[6.16,9.42],
[6.07,8.56],
[6.06,7.08],
[6.13,5.58],
[6.02,4.44],
[4.19,3.87],
[2.98,3.92],
[1.36,3.75],
# [1.08,3.88],
[0.43,3.15],
[0.31,1.63],
[0.88,0.432],
[2.49,0.0681],
[4.09,0.0833],
[5.76,0.0651],
[7.49,-0.0135],
[9.15,0.554]
])

pts = pts.T
x,y = pts
tck, u = interpolate.splprep([x, y], s=0, per=True)

xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

xi = xi.reshape((-1,1))
yi = yi.reshape((-1,1))

new_pts = np.hstack((xi,yi)).T



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
        self.waypoints = new_pts

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
        self.vis_msg.pose.orientation.w      =       1.0
        self.vis_msg.lifetime                =       Duration()
        self.count=20000
        self.position = [0,0]
        self.ldist= 0
        self.rdist= 0
        self.start_rec= None
        for i in range(self.waypoints.shape[1]):
            p       =       Point()
            p.x     =       self.waypoints[0,i]
            p.y     =       self.waypoints[1,i]
            p.z     =       0.0
            self.vis_msg.points.append(p)
        
        self.data= []
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.visualize_pub.publish(self.vis_msg)
        lidarscan_topic = '/scan'
        odomTopic = "/ego_racecar/odom"
        self.drivePub = self.create_publisher(AckermannDriveStamped,"drive",0)
        self.newscan = self.create_publisher(LaserScan,'/revised_scan', 0)
        self.odomSub = self.create_subscription(Odometry,odomTopic,self.pose_callback,0)
        self.scan_sub_ = self.create_subscription(LaserScan,lidarscan_topic,self.scan_callback,1)
        self.ld = 0.7 #lookahead distance constant to 0.5m


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here
        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
        """
        angle_increment = scan_msg.angle_increment 
        ranges = np.array(scan_msg.ranges)
        self.ldist = ranges[int(np.radians(225)/angle_increment)]
        self.rdist = ranges[int(np.radians(45)/angle_increment)]
        self.count= self.count-1
        if self.count>19004 or self.count<=0:
            print("Not recording", self.count)
            self.start_rec = [self.position[0], self.position[1]]

        if self.count>0 and self.count<=19000:
            print("recording", self.count)
            self.data.append([self.position[0], self.position[1], self.ldist, self.rdist])

        if self.count < 18000 and (np.sqrt(((np.array(self.start_rec) - np.array([self.position[0], self.position[1]])) ** 2).sum())) < 0.03:
            self.count=0

        if self.count==0:
            np.save("/sim_ws/src/pure_pursuit/scripts/Trajectory_unclipped.npy", np.array(self.data))

        print(np.sqrt(((np.array(self.start_rec) - np.array([self.position[0], self.position[1]])) ** 2).sum()))
        # print(len(self.data))
        # print(self.position, self.ldist, self.rdist)
        rev_scan = LaserScan()
        msg = np.zeros((ranges.shape))
        msg[int(np.radians(225)/angle_increment)]= ranges[int(np.radians(225)/angle_increment)]
        msg[int(np.radians(45)/angle_increment)]= ranges[int(np.radians(45)/angle_increment)]
        rev_scan=scan_msg
        rev_scan.ranges= list(msg.astype(np.float))
        self.newscan.publish(rev_scan)



    def timer_callback(self):
        self.visualize_pub.publish(self.vis_msg)


    def pose_callback(self, pose_msg):
        currPosex = pose_msg.pose.pose.position.x
        currPosey = pose_msg.pose.pose.position.y
        self.position= [currPosex, currPosey]
        currPose = np.array([currPosex,currPosey,0]).reshape((3,-1))
        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        rot_car_world = R.from_quat([qx,qy,qz,qw])       
        roll,pitch,yaw = rot_car_world.as_euler('xyz',degrees=False)
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
        goalPt = (out_pt + in_pt)/2
        x = goalPt[0]
        y = goalPt[1]
        curvature = 2*goalPt[1]/(self.ld**2)
        msg = AckermannDriveStamped()
        msg.drive.speed = float(0.5)
        msg.drive.steering_angle = curvature
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
