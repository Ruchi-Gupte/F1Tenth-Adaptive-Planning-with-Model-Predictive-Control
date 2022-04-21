import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

filename = 'Trajectory_'
data= np.load(filename+'unclipped.npy')


df = pd.DataFrame(data, columns = ['x_m','y_m','w_tr_left_m', 'w_tr_right_m'])
df.to_csv("Trajectory_unclipped.csv")


df['w_tr_left_m']= df["w_tr_left_m"].clip(0, 0.3, axis=0)
df['w_tr_right_m']= df["w_tr_right_m"].clip(0, 0.3, axis=0)
df.to_csv("Trajectory_clipped.csv")