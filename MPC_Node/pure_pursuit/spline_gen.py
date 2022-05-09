#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev
import pdb
# from turtle import pd
# import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev
import pdb
import copy
import matplotlib.pyplot as plt

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 50  # Disc pts
# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time
# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param
TARGET_SPEED = 3.0     # [m/s] target speed
N_IND_SEARCH = 10       # Search index number
DT = 0.02  # [s] time tick
# # Vehicle parameters
# LENGTH = 0.48  # [m]
# WIDTH = 0.268  # [m]
# BACKTOWHEEL = 0.1  # [m]
# WHEEL_LEN = 0.1  # [m]
# WHEEL_WIDTH = 0.05  # [m]
# TREAD = 0.5  # [m]
# WB =0.32  # [m]
# Vehicle parameters
LENGTH = 0.48  # [m]
WIDTH = 0.268  # [m]
BACKTOWHEEL = 0.07  # [m]
WHEEL_LEN = 0.07  # [m]
WHEEL_WIDTH = 0.07  # [m]
TREAD = 0.5  # [m]
WB = 0.32  # [m]
MAX_STEER = np.deg2rad(50.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(10.0)  # maximum steering speed [rad/s]
MAX_SPEED = 10.0  # maximum speed [m/s]
MIN_SPEED = 0  # minimum speed [m/s]
MAX_ACCEL = 2.5  # maximum accel [m/ss]


###Obstacle avoidance constants
DIST_FROM_OBS = 0.8 #Distance to consider in lidar for planning [m]
ANGLE_OBS     = np.deg2rad(5) 
BUBBLE        = 0.2 #This is the circle which we have to avoid in MPC


class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def update_state(state, a, delta):
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER
    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT
    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED
    return state


def cvtXYtoPolar(pt_car):
    '''
    This gets r,theta wrt x,y. Needs pt wrt car
    pt_car : x,y array of shape(2,1)
    Returns: vector of shape (2,1) polar with respect to car
    '''
    x                       =       pt_car[0]
    y                       =       pt_car[1]
    return(np.array([np.linalg.norm(pt_car),np.arctan2(y,x)]).reshape((2,1)))

# disc_del                                =   np.linspace(-MAX_STEER,MAX_STEER,15)
# disc_del                                =   np.append(disc_del,0)
# disc_vel                                =   np.arange(1,3,1)
disc_vel                                =     np.array([1, 1.5,  2])

a                                       =   0
spline_num                              =   0

splines                                 =   np.zeros((33,T+1,6)) #x,y,yaw,r,theta
for v in disc_vel:          

    if(v==1):
        disc_del                        =   np.linspace(np.deg2rad(-30.0),np.deg2rad(30.0),13)
        disc_del                        =   np.append(disc_del,0)
        DT                              =   0.02

    elif(v==2):
        disc_del                        =   np.linspace(np.deg2rad(-15.0),np.deg2rad(15.0),7)
        disc_del                        =   np.append(disc_del,0)
        DT                              =   0.02

    elif(v==1.5):
        disc_del                        =   np.linspace(np.deg2rad(-5.0),np.deg2rad(5.0),10)
        disc_del                        =   np.append(disc_del,0)
        DT                              =   0.02


        
    for d in disc_del:          
        spline_state                    =   State(v=v)
        
        print(spline_num)

        for t in range(T+1):
            spline_state                =   update_state(spline_state,a,d)


            splines[spline_num][t][0]   =   spline_state.x
            splines[spline_num][t][1]   =   spline_state.y
            splines[spline_num][t][2]   =   spline_state.yaw
            splines[spline_num][t][3]   =   spline_state.yaw



            pts                         =   np.array([spline_state.x,spline_state.y])
            r                           =   cvtXYtoPolar(pts)[0].item()
            theta                       =   cvtXYtoPolar(pts)[1].item()

            splines[spline_num][t][4]   =   r
            splines[spline_num][t][5]   =   theta
        
        spline_num+=1

        
print("Done")
print(splines.shape)
plt.plot(splines[:,:,0], splines[:,:,1])
plt.show()
np.save("spline_trajs.npy",splines)



