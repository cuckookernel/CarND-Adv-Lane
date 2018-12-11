# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:49:04 2018

@author: mrestrepo
"""
import cv2
import numpy as np

import util as u

def get_lane_pixels( rgb_correct ) :
    """Take a (perspective corrected) image and produce a binary image
      highlighting the lane pixels"""

    sobel_kernel=7

    hls = cv2.cvtColor(rgb_correct, cv2.COLOR_RGB2HLS)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    #gray = u.rgb2gray( orig )

    sobelx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=sobel_kernel)

    abs_sob_x = np.abs( sobelx )

    scaled_x = np.uint8( 255 * abs_sob_x / abs_sob_x.max() )

    mag = np.sqrt( sobelx ** 2  + sobely ** 2 + 1e-8)
    mag_scaled = np.uint8( mag / ( mag.max() / 255.0 ) )


    binary = ( ( scaled_x > 15 ) & ( mag_scaled > 21.5 )
              | ( scaled_x > 9.5) & (s_channel > 128  )
              | ( s_channel > 153) & (mag_scaled > 2.5) )


    return binary

def construct_trapezoid( img, top_offset = 50, alpha_top=1.0, alpha_bottom=1.0) :
    """Calculate the corners of a trapezoid with base along the lower edge of the image
    top side paralell to it and sides coinciding with the image diagonals.
    This assumes the camera is positioned so that top"""


    base_y = img.shape[0]

    half_height = img.shape[0] // 2
    half_width = img.shape[1] // 2

    base_left_x = int( half_width * ( 1 -  alpha_bottom ) )
    base_right_x  = int( half_width * ( 1 + alpha_bottom ) )

    # use triangle similarity
    top_half_length = int( (top_offset / half_height) * alpha_top * half_width )

    top_y  =  half_height + top_offset
    top_left_x = half_width - top_half_length
    top_right_x = half_width + top_half_length

    corners  =  [ (base_left_x, base_y),
                 ( base_right_x, base_y),
                 ( top_right_x, top_y ),
                 ( top_left_x, top_y ),
                ] #, dtype=np.int16 )

    return corners

def get_perspective_transform( img, corners, offset_x=100 ) :
    """get perspective transform"""
    dst_points  =   [ ( offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, 5 ),
                      ( offset_x, 5 ) ]

    dst = np.float32( dst_points )
    corners_nd = np.array( corners, dtype=np.float32 )
    M = cv2.getPerspectiveTransform( corners_nd, dst)

    return M, dst_points

def warp_perspective( und, M, flags=cv2.INTER_LINEAR) :

    return cv2.warpPerspective(und,  M, (und.shape[1], und.shape[0]),
                               flags=flags)


def draw_four_corners( imgc, corners, color = (255, 255, 0), tkn=3 ) :
    """Draw four segments  defined by four corner in a given order, using a
      color and thickness tkn """
    cv2.line( imgc, corners[0], corners[1], color, tkn )
    cv2.line( imgc, corners[1], corners[2], color, tkn )
    cv2.line( imgc, corners[2], corners[3], color, tkn )
    cv2.line( imgc, corners[3], corners[0], color, tkn )


# window settings
# window_width = 50
# window_height = 80 # Break image into 9 vertical layers since image height is 720
# margin = 100 # How much to slide left and right for searching



def window_mask(width, height, img_ref, center, level) :
    """ This function was taken from: Lesson 6 : Section7 - Another sliding window search
    """

    output = np.zeros_like( img_ref )
    output[int(img_ref.shape[0]-(level + 1) * height) : int(img_ref.shape[0]-level * height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output



def sliding_window_search(image, win_w, win_h, margin) :
    """ The code for this function borrows heavily from
    find_window_centroids from: Lesson 6 : Section7 - Another sliding window search.

    It just adds two return values ret_left, ret_right which are binary images
    with the same shapes as image and which will contain 1 in the pixels
    determined to be part of the left and right lanes respectively
    """

    window_centroids = [] # Store the (left,right) window centroid positions per level
    ret_left  = np.zeros( image.shape, dtype=np.bool )
    ret_right = np.zeros( image.shape, dtype=np.bool )

    window = np.ones( win_w ) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    img_h, img_w = image.shape[0], image.shape[1]
    h_win_w = win_w // 2
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * img_h / 4) : , : img_w // 2 ], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum) ) - h_win_w

    r_sum = np.sum(image[int(3 * img_h / 4) : , img_w // 2 : ], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum) ) - h_win_w +  img_w // 2

    # Add what we found for the first layer
    window_centroids.append( (l_center, r_center) )
    put_points_level( ret_left, ret_right, image, h_win_w, win_h,
                      l_center, r_center, level=0 )

    # Go through each layer looking for max pixel locations
    for level in range(1, int(img_h / win_h)):
    	   # convolve the window into the vertical slice of the image
         image_layer = np.sum(image[img_h - (level+1) * win_h :
                                    img_h - level * win_h, : ], axis=0)
         conv_signal = np.convolve(window, image_layer)
         # Find the best left centroid by using past left center as a reference
         # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

         l_min_index = max(l_center + h_win_w - margin, 0)
         l_max_index = min(l_center + h_win_w + margin, img_w)
         l_center = np.argmax(conv_signal[l_min_index : l_max_index]) + l_min_index - h_win_w
         # Find the best right centroid by using past right center as a reference
         r_min_index = max(r_center + h_win_w - margin, 0)
         r_max_index = min(r_center + h_win_w + margin, img_w)
         r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index - h_win_w
         # Add what we found for that layer
         window_centroids.append( (l_center, r_center) )

         # print( "sliding_window_search: level=%d" % level, l_center, r_center )
         put_points_level( ret_left, ret_right, image, h_win_w, win_h,
                           l_center, r_center, level )

    return window_centroids, ret_left, ret_right


def put_points_level( ret_left, ret_right, warped, h_win_w, win_h,
                      l_center, r_center, level ) :

    img_h = warped.shape[0]

    slice_v  = slice( img_h - (level + 1) * win_h , img_h - level * win_h )

    slice_hl = slice( l_center - h_win_w, l_center + h_win_w )
    ret_left[ slice_v , slice_hl ] = ( warped[ slice_v, slice_hl ] > 0 )

    slice_hr = slice( r_center - h_win_w, r_center + h_win_w )
    ret_right[ slice_v , slice_hr ] = ( warped[ slice_v, slice_hr ] > 0 )

    #print( slice_hl, slice_hr )


def show_window_search_results( warped, window_width, window_height, margin ) :
    """ The code for this function encapsulates the top level code
    from: Lesson 6 : Section7 - Another sliding window search
    """
    window_centroids, _, _ = sliding_window_search(warped, window_width,
                                                   window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like( warped )
        r_points = np.zeros_like( warped )

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	       l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
    	       r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            #Add graphic points from window mask here to total pixels found
    	       l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	       r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array( cv2.merge((zero_channel,template,zero_channel)),
                             dtype=np.uint8 ) # make window pixels green
        warpage= np.dstack( (warped, warped, warped) ) * 255 # making the original road pixels 3 color channels
        output = cv2.addWeighted( warpage, 1, template, 0.5, 0.0 ) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)


    return output


def fit_polys0( left_lane, right_lane ) :
    """Fit polinomials given separate left_lane and right_lane binary images """
    lefty, leftx   = left_lane .nonzero()
    righty, rightx = right_lane.nonzero()

    left_fit  = np.polyfit( lefty,  leftx,  deg=2 )
    right_fit = np.polyfit( righty, rightx, deg=2 )

    return left_fit, right_fit


def fit_new_polys(bin_warp, left_fit, right_fit, margin=100 ):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!

    # Grab activated pixels
    nonzeroy, nonzerox = bin_warp.nonzero()

    #yvals =  np.linspace(0, img_h - 1, img_h) # previously was nonzeroy....!

    left_curve  = np.polyval( left_fit , nonzeroy )
    right_curve = np.polyval( right_fit, nonzeroy )

    left_lane_inds  = ( (nonzerox < left_curve  + margin) & ( nonzerox >= left_curve  - margin ) ).nonzero()[0]
    right_lane_inds = ( (nonzerox < right_curve + margin) & ( nonzerox >= right_curve - margin ) ).nonzero()[0]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit  = np.polyfit( lefty,  leftx,  deg=2 )
    right_fit = np.polyfit( righty, rightx, deg=2 )

    return left_fit, right_fit


def draw_search_windows( bin_warp, left_fit, right_fit, margin ) : #,  left_lane_inds, right_lane_inds,   ) :
## Visualization ##
    # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = u.binary2rgb( bin_warp )
    window_img = np.zeros_like(out_img)

    img_h = bin_warp.shape[0]

    nonzeroy, nonzerox = bin_warp.nonzero()
    # Color in left and right line pixels

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ploty = np.linspace(0, img_h - 1, img_h )
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx  = np.polyval( left_fit, ploty )  # left_fit[0] + left_fit[1] * ploty + left_fit[2] * ploty**2
    right_fitx = np.polyval( right_fit, ploty )

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx , ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    return result

