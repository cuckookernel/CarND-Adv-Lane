# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:32:35 2018

@author: mrestrepo
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%

def test() :
    #%%

    from  common import DATA_DIR

    # Read in an image
    image = mpimg.imread( DATA_DIR + 'signs_vehicles_xygrad.png')

    ksize = 5 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(100,1000))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(100,1000))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi*1/8, np.pi/3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    print( combined.min(), combined.max() )
    plt.figure( figsize = (16,9) )

    plt.imshow( combined, cmap='gray')
    #%%


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255) ) :

    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    #%%
    if orient == 'x' :
        sobel = cv2.Sobel(gray, cv2.CV_32F,  1, 0, ksize=sobel_kernel)
    else :
        sobel = cv2.Sobel(gray, cv2.CV_32F,  0, 1, ksize=sobel_kernel)

    abs_sobel =  np.abs( sobel )
    scaled_sobel = np.uint8( 255 * abs_sobel / abs_sobel.max() )
    #%%
    binary_out = np.zeros_like( scaled_sobel )
    binary_out[ (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])  ] = 1
    #binary_out = np.copy(img) # Remove this line
    #%%
    return binary_out



def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt( sobelx ** 2  + sobely ** 2 )

    mag_scaled = np.uint8( mag / ( mag.max() / 255.0 ) )

    binary_out = np.zeros_like( mag )
    binary_out[ (mag_scaled >= mag_thresh[0]) & (mag_scaled <= mag_thresh[1])  ] = 1

    return binary_out


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    abs_sob_x = np.abs( sobelx )
    abs_sob_y = np.abs( sobely )

    grad_dir = np.arctan2(abs_sob_y, abs_sob_x)

    binary_out = np.zeros_like( grad_dir )

    binary_out[ (grad_dir >= thresh[0]) & (grad_dir <= thresh[1]) ] = 1
    return binary_out

