# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:48:38 2018

@author: mrestrepo
"""

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import cv2

from common import DATA_DIR
#%%
# Read in an image, you can also try test1.jpg or test4.jpg
image = cv2.imread( DATA_DIR + '/colorspace_test_images/test6.jpg')

#image = mpimg.imread(DATA_DIR + '/colorspace_test_images/test6.png')
#%%
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(image, thresh=(0, 255)):
    # 1) Convert to HLS color space
    #%%
    hls = cv2.cvtColor( image, cv2.COLOR_BGR2HLS)
    #%%
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like( hls )
    print( hls.shape, hls.min(), hls.max(), hls.mean()  )
    binary_output[ (hls[...,2] >= thresh[0]) & (hls[...,2] <= thresh[1])  ] = 1
    # 3) Return a binary image of threshold result
    #%%
    print( binary_output.sum() )
    return binary_output

thresh=(100, 255)
hls_binary = hls_select(image,  thresh)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))
f.tight_layout()
ax1.imshow( cv2.cvtColor( image, cv2.COLOR_BGR2RGB ) )
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary *255, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)