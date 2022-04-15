from turtle import pd
import numpy as np
import cv2
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt

import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev


threshold = 10.0

# Load your trimmed image as greyscale
image = cv2.imread("./my_smooth_map.pgm", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("./wuChen2.pgm", cv2.IMREAD_GRAYSCALE)

# Find medial axis
skeleton, distance = medial_axis(image, return_distance = True)

mask = distance < threshold

skeleton[mask] = False
skeleton = skeleton.astype(np.uint8)

# Save
cv2.imwrite("result" + str(threshold) + ".png", skeleton*255)

x,y = np.where(skeleton)

tck, u = interpolate.splprep([x, y], s=0)

xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

plt.plot(xi,yi)
plt.show()
# z = np.polyfit(x,y,2)
# f = np.poly1d(z)

# t = np.arange(0, skeleton.shape[1], 1)
# plt.figure(2, figsize=(8, 16))
# ax1 = plt.subplot(211)
# ax1.imshow(skeleton,cmap = 'gray')
# ax2 = plt.subplot(212)
# ax2.axis([0, skeleton.shape[1], skeleton.shape[0], 0])
# ax2.plot(t, f(t))
# plt.show()