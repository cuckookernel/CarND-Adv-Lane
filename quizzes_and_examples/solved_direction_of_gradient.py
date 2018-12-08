# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:46:04 2018

@author: mrestrepo
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pickle

from  common import DATA_DIR

# Read in an image
image = mpimg.imread( DATA_DIR + 'signs_vehicles_xygrad.png')

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    abs_sob_x = np.abs( sobelx )
    abs_sob_y = np.abs( sobely )

    grad_dir = np.arctan2(abs_sob_y, abs_sob_x)

    binary_out = np.zeros_like( grad_dir )

    binary_out[ (grad_dir >= thresh[0]) & (grad_dir <= thresh[1]) ] = 1
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_out = np.copy(image) # Remove this line
    return binary_out

# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)