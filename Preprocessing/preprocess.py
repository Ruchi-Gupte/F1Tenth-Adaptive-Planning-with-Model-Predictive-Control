import numpy as np
import cv2
import shutil
from turtle import pd
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev
from heapq import heappush, heappop  # Recommended.
from itertools import product


send_path 	= '/home/ruchi/f1tenth_gym_ros/maps/'
filename	= 'outMap'
sensitivity = 15
threshold = 9.0
k_size = 5
kernel = np.ones((k_size,k_size),np.uint8)

im 			= cv2.imread('./input_map/'+filename+'.pgm')
imgray 		= cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
mask_white 	= cv2.inRange(imgray, lower_white, upper_white)
blur 		= cv2.medianBlur(mask_white, 3)
smooth 		= cv2.addWeighted( mask_white, 5, mask_white, -0.5, 0)
opened = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)

cv2.imwrite(send_path+filename+".pgm", opened)
cv2.imwrite(filename+".pgm", opened)
shutil.copyfile(filename+".yaml", send_path+filename+".yaml")

# Find medial axis
skeleton, distance = medial_axis(opened, return_distance = True)
mask = distance < threshold
skeleton[mask] = False
skeleton = skeleton.astype(np.uint8)*255
cv2.imwrite("centreline_result_" + str(threshold) + ".png", skeleton)














#Djikstras too annoying omg
# x,y= np.where(skeleton)
# nodelist= np.hstack((x.reshape(len(x),1),y.reshape(len(x),1)))
# maxlen=3
# for i in range(len(x)):
#     start=np.array(nodelist[i])
#     visited_nodes = [(list(start), list(start), 0)]
#     seen = set()
#     seen.add(tuple(list(nodelist[i])))
#     flag= 0
#     for relative_val in product((-1, 0, 1), repeat=2):
#         a = start + np.array(relative_val)
#         if a in nodelist and tuple(a) not in seen and flag==0:    
#             seen.add(tuple(list(a)))
#             for relative_val in product((-1, 0, 1), repeat=2):
#                 b = a + np.array(relative_val)
#                 if b in nodelist and tuple(b) not in seen:   
#                     goal = b 
#                     flag=1
#                     break
#         if flag==1:
#             break   
#     if flag==0:
#         break
    

#     current_nodes = [(0, list(start))]
#     while len(current_nodes) > 0:
#         d, node = heappop(current_nodes)
        
#         xdif= (x.reshape(len(x),1) - np.ones((len(x),1))*node[0])
       
#         ydif= (y.reshape(len(x),1) - np.ones((len(x),1))*node[1])
#         dis = np.where( xdif+ydif <=2, xdif,ydif)
#         print(dis)
#         rr
#             # if (tuple(nei) not in seen) and (dis==1 or dis==np.sqrt(2)) and count<4:
#             #     count=count+1
#             #     heappush(current_nodes, (d + 1, list(a)))
#             #     seen.add(tuple(list(a)))
#             #     visited_nodes.append([node, list(a), d+1])
#             #     if list(a) == list(goal):
#             #         print("OMG", len(visited_nodes))
            
#         #             node = visited_nodes[-1]
#         #             depth = node[2]
#         #             while depth != 2:
#         #                 prev_node = node[0]
#         #                 depth = node[2]
#         #                 path.append(occ_map.index_to_metric_center(prev_node))
#         #                 for n in visited_nodes:
#         #                     fin_d = np.inf
#         #                     if n[2] == depth - 1 and tuple(n[1]) == tuple(prev_node):
#         #                         if fin_d > n[3]:
#         #                             find_d = n[3]
#         #                             temp = n
#         #                 node = temp
#         #             path.append(start)
#         #             path.reverse()
#         #             path = np.array(path)
#         #             return path, nodes_expanded	