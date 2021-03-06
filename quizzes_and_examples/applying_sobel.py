# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:09:19 2018

@author: mrestrepo
"""

import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pickle

sys.path.append( './')
#%%
from common import DATA_DIR, greys_cmap



# Read in an image and grayscale it
#%%
# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply the following steps to img
    # 1) Convert to grayscale
    #%%
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    #%%
    if orient == 'x' :
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    else :
        sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    abs_sobel =  np.abs( sobel )
    scaled_sobel = np.uint8( 255 * abs_sobel / abs_sobel.max() )
    #%%
    plt.figure( figsize=(16,9))
    plt.imshow( 255 - abs_sobel, cmap=greys_cmap )
    #%%
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_out = np.zeros_like( scaled_sobel )
    binary_out[ (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)  ] = 1
    #binary_out = np.copy(img) # Remove this line
    #%%
    return binary_out


def test() :
    #%%
    # Run the function
    orient='x'
    #%%
    img = mpimg.imread(DATA_DIR + 'signs_vehicles_xygrad.png')

    grad_binary = abs_sobel_thresh(img, orient=orient, thresh_min=20, thresh_max=100)
    #| Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #%%