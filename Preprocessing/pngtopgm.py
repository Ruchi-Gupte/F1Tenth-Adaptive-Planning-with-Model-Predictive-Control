import numpy as np
import cv2
import shutil
from PIL import Image
from turtle import pd
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev
from heapq import heappush, heappop  # Recommended.
from itertools import product


filename	= 'outMap'

im 			= cv2.imread(filename+'.pgm')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im2 			= cv2.imread('centerline_clean.png')
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)




im3=im-im2

ulta= cv2.bitwise_not(im2) 

cv2.imwrite("centrline.pgm", im2)
cv2.imwrite("overlay.pgm", im3)
cv2.imwrite("ulta_centerline.pgm", ulta )
