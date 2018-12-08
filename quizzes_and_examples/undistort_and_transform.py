# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:33:32 2018

@author: mrestrepo
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import DATA_DIR

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( DATA_DIR + "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread( DATA_DIR + 'correct_for_distortion/test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y
#%%
# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    #%%
    und = cv2.undistort(img, mtx, dist, None, mtx)
    #%%
    # 2) Convert to grayscale
    gray = cv2.cvtColor( und, cv2.COLOR_BGR2GRAY )
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    #img1 = cv2.drawChessboardCorners( und.copy(), (nx,ny), corners, ret )
    #%%
    four_corners = corners[[0, nx-1, -nx, -1],...].squeeze()
    #%%
    img2 = cv2.drawChessboardCorners( und.copy(), (nx,ny), four_corners, ret )
    #%%
    plt.imshow( img2 )
    #%%

    #x0,y0 = four_corners[0,0], four_corners[0,1]
    x0,y0 = 50,50

    x1,y1 = x0 + 150 * nx, y0 + 150 * ny # four_corners[3,0], four_corners[3,1]
    dst = np.float32( [[x0,y0], [x1,y0],[x0,y1], [x1,y1]] )
    # 4) If corners found:

            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    M = cv2.getPerspectiveTransform(four_corners, dst)

    #warped = np.copy(img)
    warped = cv2.warpPerspective(und, M, (x1+50, y1+50), flags=cv2.INTER_LINEAR)
    plt.imshow( warped )
    #%%
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
