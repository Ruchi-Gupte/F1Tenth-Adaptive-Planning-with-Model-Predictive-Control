#!/usr/bin/env python3

#  ROS Imports 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseStamped
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry

# Python Imports
import numpy as np
import pdb
import copy
from threading import local
import cvxpy
import math
import sys
import os
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as Rot

NX                      =       4                               # x = x, y, v, yaw
NU                      =       2                               # a = [accel, steer]
T                       =       3                               # horizon length

# MPC parameters
R                       =       np.diag([0.01, 0.01])           # input cost matrix
Rd                      =       np.diag([0.01, 1.0])            # input difference cost matrix
Q                       =       np.diag([1.0, 1.0, 0.5, 0.5])   # state cost matrix
Qf                      =       Q                               # state final matrix
GOAL_DIS                =       1.5                             # goal distance
STOP_SPEED              =       0.5 / 3.6                       # stop speed
MAX_TIME                =       500.0                           # max simulation time

# Iterative paramter
MAX_ITER                =       3                               # Max iteration
DU_TH                   =       0.1                             # iteration finish param
TARGET_SPEED            =       3.0                             # [m/s] target speed
N_IND_SEARCH            =       10                              # Search index number
DT                      =       0.1                             # [s] time tick

# Vehicle parameters
LENGTH                  =       0.48                            # [m] 
WIDTH                   =       0.268                           # [m] 
BACKTOWHEEL             =       0.07                            # [m] 0.1
WHEEL_LEN               =       0.07                            # [m] 0.1
WHEEL_WIDTH             =       0.07                            # [m]  0.05
TREAD                   =       0.5                             # [m]
WB                      =       0.32                            # [m]
MAX_STEER               =       np.deg2rad(20.0)                # maximum steering angle [rad]
MAX_DSTEER              =       np.deg2rad(5.0)                 # maximum steering speed [rad/s]
MAX_SPEED               =       10.0                            # maximum speed [m/s]
MIN_SPEED               =       0                               # minimum speed [m/s]
MAX_ACCEL               =       2.5                             # maximum accel [m/ss]

class State:
    """
    Vehicle State Class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

def cvtXYtoPolar(pt_car):
    '''
    This gets r,theta wrt x,y. Needs pt wrt car
    pt_car : x,y array of shape(2,1)
    Returns: vector of shape (2,1) polar with respect to car
    '''
    x                       =       pt_car[0]
    y                       =       pt_car[1]
    return(np.array([np.linalg.norm(pt_car),np.arctan2(y,x)]).reshape((2,1)))

class MPC(Node):
    """ 
    Implement MPC on the car
    """
    def __init__(self):
        super().__init__('mpc_node')
        """
            self.waypoints: Load the waypoint saved 
            Type: numpy array -> Shape : [2,1000] where 2 corresponds to x and y and 1000 are the number of points
        """
        traj                    =       np.load("/sim_ws/src/pure_pursuit/trajectory22April.npy")
        self.waypoints          =       traj

        """
        local_traj_path:
                    Shape: (Num_trajectories, num_points, 5)
                    5-> x,y,yaw,v,r,theta 
        """
        local_traj_path         =       np.load("/sim_ws/src/pure_pursuit/spline_trajs.npy")
        self.local_path         =       local_traj_path
        

        """
        Visualisations:
        local_viz_topic  : Potential Trajectories for Local Planning
        global_path_topic: TUMN Global optimized trajectory / Reference Trajectory
        horizon_topic    : Horizon points
        """
        local_viz_topic         =        "local_viz"
        self.local_path_viz     =       self.create_publisher(Marker, local_viz_topic, 10)
        
        global_path_topic                    =       "global_path_viz"
        self.visualize_pub                   =       self.create_publisher(Marker, global_path_topic, 10)

        horizon_topic                        =       "horizon_viz"
        self.visualize_pub2                  =       self.create_publisher(Marker, horizon_topic, 10)
        
        self.glob_vis_msg                         =       Marker()
        self.glob_vis_msg.header.frame_id         =       "/map"
        self.glob_vis_msg.type                    =       Marker.POINTS
        self.glob_vis_msg.action                  =       Marker.ADD
        self.glob_vis_msg.scale.x                 =       0.1
        self.glob_vis_msg.scale.y                 =       0.1
        self.glob_vis_msg.scale.z                 =       0.1
        self.glob_vis_msg.color.g                 =       1.0
        self.glob_vis_msg.color.a                 =       1.0
        self.glob_vis_msg.pose.orientation.w      =       1.0
        # self.glob_vis_msg.lifetime                =       Duration()
        self.glob_vis_msg.lifetime.nanosec        =       30  
      
        self.cx                              =       traj[:,0].tolist()
        self.cy                              =       traj[:,1].tolist()
        self.sp                              =       traj[:,2].tolist()
        self.cyaw                            =       (np.deg2rad(90) + traj[:,3]) # self.cyaw%(2*math.pi)
        self.cyaw[self.cyaw<0]               =       self.cyaw[self.cyaw<0] + 2*math.pi 
        self.cyaw                            =       self.cyaw.tolist()
        self.ck                              =       traj[:,4].tolist()       
        
        # for i in range(1):
        for i in range(self.waypoints.shape[0]):
            p       =       Point()
            p.x     =       self.waypoints[i,0]
            p.y     =       self.waypoints[i,1]
            p.z     =       0.0
            self.glob_vis_msg.points.append(p)
        
        self.horzn_vis_msg                  =       Marker()
        self.horzn_vis_msg                  =       copy.deepcopy(self.glob_vis_msg)
        self.horzn_vis_msg.points           =       []
        self.horzn_vis_msg.color.g          =       0.0
        self.horzn_vis_msg.color.r          =       1.0

        self.local_vis_msg                  =       Marker()
        self.local_vis_msg                  =       copy.deepcopy(self.glob_vis_msg)
        self.local_vis_msg.points           =       []
        self.local_vis_msg.color.g          =       0.0
        self.local_vis_msg.color.r          =       0.0
        self.local_vis_msg.color.b          =       0.0

        self.visualize_pub.publish(self.glob_vis_msg)

        """
        Subsctibers:
            odomTopic    : Odometry
            scanTopic    : LaserScan

        Publishers:
            drivePub     : Publish AckermanDriveStamped to car
            OppDrivePub  : Publish AckermanDriveStamped to opponent car
        """
        odomTopic           =      "/ego_racecar/odom"
        self.odomSub        =      self.create_subscription(Odometry,odomTopic,self.pose_callback,0)

        scanTopic           =       '/scan'
        self.lidarSub       =       self.create_subscription(LaserScan,scanTopic,self.lidar_callback,10)

        self.drivePub       =       self.create_publisher(AckermannDriveStamped,"drive",0)
        self.OppDrivePub    =       self.create_publisher(AckermannDriveStamped,"/opp_drive",0)
        
        # Initialized Parameters
        self.angleMax               =       0
        self.step                   =       0
        self.ranges                 =       0
        self.islidaron              =       False

        self.initialize             =       True
        self.old_input              =       0
        self.minRange               =       0.11999999731779099
        self.maxRange               =       20.0
        self.disparity_thresh_min   =       1

    def lidar_callback(self, data):
        """ 
        Process each LiDAR scan and Apply Disparity Extender algorithm
        """
        self.step                           =       data.angle_increment
        self.angleMax                       =       data.angle_max
        self.ranges                         =       np.array(data.ranges)
        self.islidaron                      =       True
        proc_ranges                         =       self.ranges
        proc_ranges[np.isinf(proc_ranges)]  =       self.maxRange  
        proc_ranges[np.isnan(proc_ranges)]  =       self.minRange
        disparity                           =       proc_ranges[:-1] - proc_ranges[1:]
        disparity_bool                      =       np.abs(disparity) >= self.disparity_thresh_min
        disparity_bool_idx                  =       np.where(disparity_bool)[0]

        for idx in disparity_bool_idx:
            min_idx                         =       max(0,idx - 30)
            max_idx                         =       min(idx + 30,proc_ranges.shape[0])
            proc_ranges[min_idx:max_idx]    =       np.min(proc_ranges[min_idx:max_idx])

        self.ranges  = proc_ranges
        
    
    
    def pose_callback(self, pose_msg):
        if not self.islidaron:
            return   
        
        dl              =   0.05
        x               =   pose_msg.pose.pose.position.x
        y               =   pose_msg.pose.pose.position.y
        currPose        =   np.array([x,y,0]).reshape((3,-1))
        qx              =   pose_msg.pose.pose.orientation.x
        qy              =   pose_msg.pose.pose.orientation.y
        qz              =   pose_msg.pose.pose.orientation.z
        qw              =   pose_msg.pose.pose.orientation.w
        rot_car_world   = Rot.from_quat([qx,qy,qz,qw])
        roll,pitch,yaw  = rot_car_world.as_euler('xyz',degrees=False)
        
        if(yaw < 0):
            yaw = yaw + 2*math.pi

        v               =   pose_msg.twist.twist.linear.x
        state           =   State(x,y,yaw,v)

        if(self.initialize):
            self.initialize         =   False
            self.target_ind         =   np.argmin(np.linalg.norm(self.waypoints[:,:2] - np.array([x,y]).reshape(1,-1),axis = 1))
            self.target_ind, _      =   self.calc_nearest_index(state, self.cx, self.cy, self.cyaw, self.target_ind)
            self.odelta, self.oa    =   None, None
        self.target_ind, _ = self.calc_nearest_index(state, self.cx, self.cy, self.cyaw, self.target_ind)

        if(self.target_ind > len(self.cx) - T):
            self.target_ind = 0
        
        # if state.yaw - self.cyaw[self.target_ind] >= math.pi:
        #     # print("I am here")
        #     state.yaw -= math.pi * 2.0
        # elif state.yaw - self.cyaw[self.target_ind] <= -math.pi:
        #     # print("I am here2")
        #     state.yaw += math.pi * 2.0
        
        self.visualize_pub.publish(self.glob_vis_msg)
        
        xref, self.target_ind, dref = self.calc_ref_trajectory(state, self.cx, self.cy, self.cyaw, self.ck, self.sp, dl, self.target_ind)

        self.horzn_vis_msg.points   = []
        
        # for i in range(1):
        for i in range(xref.shape[1]):
            p       =       Point()
            p.x     =       xref[0,i]
            p.y     =       xref[1,i]
            p.z     =       0.0
            self.horzn_vis_msg.points.append(p)
        self.visualize_pub2.publish(self.horzn_vis_msg)

        dummy_zeros                         =           np.zeros(self.local_path.shape[1]).reshape(-1,1)
        
        #Shape: 101,3
        self.local_vis_msg.points           =           []
        self.local_vis_msg.colors           =           []
        
        best_trajectory_number              =           0
        best_closest_idx                    =           0

        #Shape: [50,2]
        num_idx_to_search       =  35
        if self.target_ind      + num_idx_to_search < self.waypoints.shape[0]:
            points_to_search    =   self.waypoints[self.target_ind : self.target_ind + num_idx_to_search,:2]

        else:
            points_to_stack     =  (self.target_ind + num_idx_to_search) - self.waypoints.shape[0]
            points_to_search    =   self.waypoints[self.target_ind : self.target_ind + num_idx_to_search,:2]
            points_to_search    =   np.vstack((points_to_search, self.waypoints[: points_to_stack,:2]))
        
        obstacle_detect= False
        
        ''' Other Obstacle Detection Method : Not as good'''
        # for points in points_to_search:
        #     ref_r, ref_th= cvtXYtoPolar(points)
        #     theta_index = int(self.angleMax/self.step) + int(ref_th/self.step)
        #     is_occ = np.max(self.ranges[theta_index] > ref_r)
        #     is_occ += np.max(self.ranges[theta_index+5] < rdist)
        #     is_occ += np.max(self.ranges[theta_index-5] < rdist)
        #     if is_occ:
        #         obstacle_detect= True
        # print(obstacle_detect)
        
        if np.min(self.ranges[int(self.angleMax/self.step)-50: int(self.angleMax/self.step)+50]) < 2.5:
            obstacle_detect= True
            print(obstacle_detect)

        if obstacle_detect:
            # for i in range(self.local_path.shape[0]-1,-1,-1):
            # for i in range(1):
            for i in range(self.local_path.shape[0]):
                self.local_path_1               =           self.local_path[i,:,:2]                             #Shape: 101,
                rdist, theta_r                  =           self.local_path[i,:,-2], self.local_path[i,:,-1]    #Shape: 101,2
                theta_index                     =           int(self.angleMax/self.step) + (theta_r/self.step).astype(int)
                is_occ                          =           np.max(self.ranges[theta_index] < rdist)
                # is_occ                       +=           np.max(self.ranges[theta_index+5] < rdist)
                # is_occ                       +=           np.max(self.ranges[theta_index-5] < rdist)
                world_local_points_1            =           rot_car_world.apply( np.hstack((self.local_path_1, dummy_zeros))) + currPose.T

                if is_occ:
                    c   = ColorRGBA()
                    c.r = 1.0
                    c.b = 0.0
                    c.g = 0.0
                    c.a = 1.0
                else:
                    c = ColorRGBA()
                    c.r = 0.0
                    c.b = 1.0
                    c.g = 0.0
                    c.a = 1.0
                    lastp               =     world_local_points_1[-1,:][:2].reshape(1,-1)    #Shape:(1,2)
                    closest_idx         =     np.argmin(np.linalg.norm(lastp - points_to_search , axis = 1))
                    if closest_idx >= best_closest_idx:
                        best_closest_idx        =   closest_idx
                        best_trajectory_number  =   i

                for j in range(self.local_path_1.shape[0]):
                    p           =       Point()
                    p.x         =       world_local_points_1[j,0]
                    p.y         =       world_local_points_1[j,1]
                    p.z         =       0.0

                    self.local_vis_msg.points.append(p)
                    self.local_vis_msg.colors.append(c)

            print(best_trajectory_number)
            self.local_path_1               =           self.local_path[best_trajectory_number,:,:2]    #Shape: 101,
            world_local_points_1            =           rot_car_world.apply( np.hstack((self.local_path_1, dummy_zeros))) + currPose.T
            spline_yaw                      =           self.local_path[best_trajectory_number,:,2] + yaw
            c   = ColorRGBA()
            c.r = 0.937
            c.b = 0.258
            c.g = 0.960
            c.a = 1.0
                
            for j in range(self.local_path_1.shape[0]):
                p           =       Point()
                p.x         =       world_local_points_1[j,0]
                p.y         =       world_local_points_1[j,1]
                p.z         =       0.0
                self.local_vis_msg.points.append(p)
                self.local_vis_msg.colors.append(c)

            self.local_path_viz.publish(self.local_vis_msg)
            """
            xref: T,4 -> x,y,v,yaw
            """
            local_x         = world_local_points_1[:,0]
            local_y         = world_local_points_1[:,1]
            idx_to_sample   = np.linspace(0, world_local_points_1.shape[0] - 1, T +1 , dtype = int)
            # idx_to_sample = np.linspace(0, 25, T +1 , dtype = int)

            local_ref       = xref
            local_ref[0,:]  = local_x[idx_to_sample]
            local_ref[1,:]  = local_y[idx_to_sample]

            local_spline_yaw                        = spline_yaw[idx_to_sample]
            local_spline_yaw[local_spline_yaw<0]    = local_spline_yaw[local_spline_yaw<0] + 2*math.pi
            
            if(abs(state.yaw - local_spline_yaw[0]) > 3.14):
                if(state.yaw < local_spline_yaw[0]):
                    state.yaw += 2*math.pi
                else:
                    print("hey you out there in the cold, ")
                    local_spline_yaw[0] += 2*math.pi

            for i in range(1,local_spline_yaw.shape[0]):
                if local_spline_yaw[i] < 1.0 and local_spline_yaw[i-1] > 6.0:
                        local_spline_yaw[i] += 2*math.pi

            for i in range(T-1,-1,-1):
                if (local_spline_yaw[i] - local_spline_yaw[i+1]) < -math.pi:
                    local_spline_yaw[i] += 2*math.pi
            
            if(state.yaw - local_spline_yaw[0]) < -math.pi:
                state.yaw += 2*math.pi

            local_ref[3,:]      =   local_spline_yaw
            xref                =   local_ref          
        self.local_path_viz.publish(self.local_vis_msg)

        x0                      =   [state.x, state.y, state.v, state.yaw]  # current state
        self.oa, self.odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(xref, x0, dref, self.oa, self.odelta)
        
        # print("s_yaw:", state.yaw, "|t_yaw:", local_spline_yaw[0], "|steer:", self.odelta[0], "|s_x:",state.x,"|s_y:",state.y,"|t_x:",self.cx[self.target_ind],"|t_y:",self.cy[self.target_ind])
        # print(local_spline_yaw)
        # print("xref: ", xref[3,:])

        if self.odelta is not None:
            # print("Publishing")
            # print("di: ", di)
            # print("ai: ", ai)
            # print("ov: ", ov[0])
            di, ai = self.odelta[0], self.oa[0]
            msg = AckermannDriveStamped()
            msg.drive.acceleration      = float(ai)
            msg.drive.steering_angle    = float(di)
            self.old_input              =   di
            # msg.drive.speed           =  float(ov[0])
            msg.drive.speed           =  float(self.sp[self.target_ind])*0.7
            # print(msg.drive.speed)
            # msg.drive.speed           =  5.0
            # msg.drive.speed             =  0.0
            self.drivePub.publish(msg)
        
            msg.drive.speed             = 0.0
            self.OppDrivePub.publish(msg)

    def get_linear_model_matrix(self,v, phi, delta):
        A           =   np.zeros((NX, NX))
        A[0, 0]     =   1.0
        A[1, 1]     =   1.0
        A[2, 2]     =   1.0
        A[3, 3]     =   1.0
        A[0, 2]     =   DT * math.cos(phi)
        A[0, 3]     =   - DT * v * math.sin(phi)
        A[1, 2]     =   DT * math.sin(phi)
        A[1, 3]     =   DT * v * math.cos(phi)
        A[3, 2]     =   DT * math.tan(delta) / WB
        B           =   np.zeros((NX, NU))
        B[2, 0]     =   DT
        B[3, 1]     =   DT * v / (WB * math.cos(delta) ** 2)
        C           =   np.zeros(NX)
        C[0]        =   DT * v * math.sin(phi) * phi
        C[1]        =   - DT * v * math.cos(phi) * phi
        C[3]        =   - DT * v * delta / (WB * math.cos(delta) ** 2)
        return A, B, C
    
    def linear_mpc_control(self,xref, xbar, x0, dref):
        """
        Linear MPC control
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
        x               = cvxpy.Variable((NX, T + 1))
        u               = cvxpy.Variable((NU, T))
        cost            = 0.0
        constraints     = []
        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)
            if t != 0:
                cost    += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
            A, B, C      = self.get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            
            if t < (T - 1):
                cost        += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]
        
        cost            +=  cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
        constraints     +=  [x[:, 0] == x0]
        constraints     +=  [x[2, :] <= MAX_SPEED]
        constraints     +=  [x[2, :] >= MIN_SPEED]
        constraints     +=  [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints     +=  [cvxpy.abs(u[1, :]) <= MAX_STEER]
        prob             =  cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox          =   self.get_nparray_from_matrix(x.value[0, :])
            oy          =   self.get_nparray_from_matrix(x.value[1, :])
            ov          =   self.get_nparray_from_matrix(x.value[2, :])
            oyaw        =   self.get_nparray_from_matrix(x.value[3, :])
            oa          =   self.get_nparray_from_matrix(u.value[0, :])
            odelta      =   self.get_nparray_from_matrix(u.value[1, :])
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
        return oa, odelta, ox, oy, oyaw, ov

    def calc_ref_trajectory(self,state, cx, cy, cyaw, ck, sp, dl, pind):
        xref        =   np.zeros((NX, T + 1))
        dref        =   np.zeros((1, T + 1))
        ncourse     =   len(cx)
        tref        =   cyaw[pind]
        
        ind, _      =   self.calc_nearest_index(state, cx, cy, cyaw, pind)
        if pind >= ind:
            ind = pind

        xref[0, 0]  =   cx[ind]
        xref[1, 0]  =   cy[ind]
        xref[2, 0]  =   sp[ind]
        xref[3, 0]  =   cyaw[ind]
        dref[0, 0]  =   0.0  # steer operational point should be 0
        travel = 0.0
        
        if(abs(state.yaw - xref[3,0]) > 3.14):
            if(state.yaw < xref[3,0]):
                state.yaw += 2*math.pi
            else:
                print("Hey you out there in the cold")
                xref[3,0] += 2*math.pi
        
        for i in range(1,T + 1):
            travel          +=  abs(state.v) * DT
            dind            =   int(round(travel / dl))
            if (ind + dind) < ncourse:
                xref[0, i]  =   cx[ind + dind]
                xref[1, i]  =   cy[ind + dind]
                xref[2, i]  =   sp[ind + dind]
                xref[3, i]  =   cyaw[ind + dind]
                dref[0, i]  =   0.0
            else:
                xref[0, i]  =   cx[ncourse - 1]
                xref[1, i]  =   cy[ncourse - 1]
                xref[2, i]  =   sp[ncourse - 1]
                xref[3, i]  =   cyaw[ncourse - 1]
                dref[0, i]  =   0.0
            if(i>0):
                if xref[3,i] < 1.0 and xref[3,i-1] > 6.0:
                    xref[3,i] += 2*math.pi
        
        for i in range(T-1,-1,-1):
            if (xref[3,i] - xref[3,i+1]) < -math.pi:
                xref[3,i] += 2*math.pi
        if(state.yaw - xref[3,0]) < -math.pi:
            state.yaw += 2*math.pi

        return xref, ind, dref

    def iterative_linear_mpc_control(self,xref, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        """
        if oa is None or od is None:
            oa                       =       [0.0] * T
            od                       =       [0.0] * T
        for i in range(MAX_ITER):
            xbar        = self.predict_motion(x0, oa, od, xref)
            poa, pod                 =       oa[:], od[:]
            oa, od, ox, oy, oyaw, ov =       self.linear_mpc_control(xref, xbar, x0, dref)
            du                       =       sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= DU_TH:
                break
        else:
            print("Iterative is max iter")
        return oa, od, ox, oy, oyaw, ov

    def predict_motion(self,x0, oa, od, xref):
        xbar             =   xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0]   =   x0[i]
        state            =   State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])

        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state        = self.update_state(state, ai, di)
            xbar[0, i]   = state.x
            xbar[1, i]   = state.y
            xbar[2, i]   = state.v
            xbar[3, i]   = state.yaw
        return xbar

    def update_state(self,state, a, delta):
        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER

        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x     =   state.x + state.v * math.cos(state.yaw) * DT
        state.y     =   state.y + state.v * math.sin(state.yaw) * DT
        state.yaw   =   state.yaw + state.v / WB * math.tan(delta) * DT
        state.v     =   state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED

        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state
        
    def check_goal(state, goal, tind, nind):
        # check goal
        dx          =   state.x - goal[0]
        dy          =   state.y - goal[1]
        d           =   math.hypot(dx, dy)
        isgoal      =   (d <= GOAL_DIS)
        if abs(tind - nind) >= 5:
            isgoal  =   False
        isstop      =   (abs(state.v) <= STOP_SPEED)
        if isgoal and isstop:
            return True
        return False

    def smooth_yaw(self,yaw):
        for i in range(len(yaw) - 1):
            dyaw    = yaw[i + 1] - yaw[i]
            while dyaw >= math.pi / 2.0:
                yaw[i + 1]  -= math.pi * 2.0
                dyaw         =  yaw[i + 1] - yaw[i]
            while dyaw <= -math.pi / 2.0:
                yaw[i + 1]  += math.pi * 2.0
                dyaw         = yaw[i + 1] - yaw[i]
        return yaw

    def get_nparray_from_matrix(self,x):
        return np.array(x).flatten()

    def calc_nearest_index(self,state, cx, cy, cyaw, pind):
        dx      =   [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy      =   [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
        d       =   [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind    =   min(d)
        ind     =   d.index(mind) + pind
        mind    =   math.sqrt(mind)
        dxl     =   cx[ind] - state.x
        dyl     =   cy[ind] - state.y
        angle   =   self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))

        if angle < 0:
            mind *= -1
        return ind, mind

    def pi_2_pi(self,angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi
        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initalized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)
    mpc_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()