# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:13:16 2018

@author: Mateo
"""
import numpy as np

from pipeline import put_points_level

def left_and_right_lane_points( warped, win_w, win_h, margin ) :

    img_h = warped.shape[0]
    h_win_w = win_w // 2

    # will populate and return the following
    ret_left  = np.zeros( warped.shape, dtype=np.bool )
    ret_right = np.zeros( warped.shape, dtype=np.bool )

    n_layers = int( img_h / win_h )

    centroids = find_window_centroids(warped, win_w, win_h, margin)

    assert len(centroids) == n_layers, \
           f"n_layers = {n_layers} != len(centroids) = {len(centroids)}"

    for level, centroid in zip( range(0, n_layers ), centroids ) :
        l_center= int(centroid[0])
        r_center= int(centroid[1])

        print( l_center, r_center )

        put_points_level( ret_left, ret_right, warped, h_win_w, win_h,
                          l_center, r_center, level )


    return ret_left, ret_right


def find_window_centroids(image, window_width, window_height, margin):
    """ This function was taken from: Lesson 6 : Section7 - Another sliding window search
    """
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones( window_width ) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    img_h, img_w = image.shape[0], image.shape[1]
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum( image[int(3 * img_h / 4) : , : int(img_w / 2) ], axis=0 )
    l_center = np.argmax(np.convolve(window, l_sum) ) - window_width / 2

    r_sum = np.sum( image[int(3 * img_h / 4) : , int(img_w / 2) : ], axis=0 )
    r_center = np.argmax(np.convolve(window, r_sum) ) - window_width/2 + int(img_w / 2)

    # Add what we found for the first layer
    window_centroids.append( (l_center, r_center) )

    # Go through each layer looking for max pixel locations
    for level in range(1, int(img_h / window_height)):
    	    # convolve the window into the vertical slice of the image
    	    image_layer = np.sum(image[int(img_h - (level+1) * window_height) :
                                    int(img_h - level * window_height), : ], axis=0)
    	    conv_signal = np.convolve(window, image_layer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = window_width/2
    	    l_min_index = int(max(l_center + offset - margin, 0))
    	    l_max_index = int(min(l_center + offset + margin, img_w))
    	    l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
    	    # Find the best right centroid by using past right center as a reference
    	    r_min_index = int(max(r_center + offset - margin, 0))
    	    r_max_index = int(min(r_center + offset + margin, img_w))
    	    r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
    	    # Add what we found for that layer
    	    window_centroids.append( (l_center, r_center) )

    return window_centroids