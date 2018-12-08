# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:11:44 2018

@author: mrestrepo
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from common import DATA_DIR, rgb_read
# Load our image
# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1

#im0 = cv2.imread( DATA_DIR + 'warped-example')
#%%
img = cv2.cvtColor( rgb_read(DATA_DIR + 'warped-example.jpg'), cv2.COLOR_RGB2GRAY )/255

#%%
def hist(img):
    #%% Lane lines are likely to be mostly vertical nearest to the car
    height = img.shape[0]
    bottom_half = img[height//2 : height, :]

    #%% TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram =  bottom_half.sum( axis=0 )

    return histogram

# Create histogram of image binary activations
histogram = hist(img)

# Visualize the resulting histogram
plt.plot(histogram)