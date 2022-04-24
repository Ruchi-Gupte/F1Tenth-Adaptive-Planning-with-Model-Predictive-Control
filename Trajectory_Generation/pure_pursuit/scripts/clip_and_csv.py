import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import shutil

filename = 'Trajectory_'
data= np.load(filename+'unclipped.npy')
send_path= '/home/ruchi/f1Tenth_project_final/Ideal_Trajectory_Velocity generation/inputs/tracks/'

df = pd.DataFrame(data, columns = ['x_m','y_m','w_tr_left_m', 'w_tr_right_m'])
df.to_csv("Trajectory_unclipped.csv", index=False)


df['w_tr_left_m']= df["w_tr_left_m"].clip(0, 0.3, axis=0)
df['w_tr_right_m']= df["w_tr_right_m"].clip(0, 0.3, axis=0)
df.to_csv("Trajectory_clipped.csv", index=False)

#shutil.copyfile(filename+"unclipped.csv", send_path+filename+"unclipped.csv")
#shutil.copyfile(filename+"clipped.csv", send_path+filename+"clipped.csv")
