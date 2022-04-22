import numpy as np

my_data = np.genfromtxt('traj_race_cl.csv', delimiter=';')
my_array  = np.vstack((my_data[:,1], my_data[:,2], my_data[:,5], my_data[:,3], my_data[:,4])).T
np.save("trajectory.npy",my_array)
