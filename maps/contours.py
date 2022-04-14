import numpy as np
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

im = cv.imread('./../lane/wuChen2.pgm')
imgray = cv.cvtColor(im, cv.COLOR_BGR2HSV)

sensitivity = 15
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
mask_white = cv.inRange(imgray, lower_white, upper_white)

blur = cv2.medianBlur(mask_white, 7)
#blur = cv.GaussianBlur( mask_white, (3,3), 0)
smooth = cv.addWeighted( mask_white, 2.5, mask_white, -0.5, 0)

contours, hierarchy = cv.findContours(smooth, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

separate_contours = [ ]
temp= []
count=0
for i in contours:
	if count==0:
		x_p= i[0][0][0]
		y_p= i[0][0][1]
		count=count+1
	x= i[0][0][0]
	y= i[0][0][1]
	
	if abs(x_p-x) + abs(y_p-y >=50):
		separate_contours.append(temp)
		temp=[]
		print("Aye")
	temp.append([x_p, y_p])
	count=count+1
	x_p= i[0][0][0]
	y_p= i[0][0][1]
print(np.array(contours))	
im = cv.imread('wuchen.pgm')
cv.drawContours(im,contours, -1,(255,0,0), 3)
print(np.array(separate_contours)[0])
cv.imwrite("wuchen.pgm", smooth)
cv.imshow('img', im)
cv.waitKey(50000)
