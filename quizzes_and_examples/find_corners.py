# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:51:20 2018

@author: mrestrepo
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#from util import rgb_read
def rgb_read( path ) :
    if not os.path.exists( path ) :
        raise RuntimeError( "File not found: "  + path )
    return cv2.imread( path )[:, :, ::-1].copy()

#%%
os.chdir("c:/Users/mrestrepo/git/CarND-Advanced-Lane-Lines") #CarND-Advcanced-Lane-Lines")
#print( os.listdir() )
# prepare object points
nx = 9 #TODO: enter the number of inside corners in x
ny = 5 #TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = 'camera_cal/calibration1.jpg'
#img1 = cv2.imread( fname )
img = rgb_read( fname )

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
