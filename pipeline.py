# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:49:04 2018

@author: mrestrepo
"""
import cv2
import numpy as np

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
    dst_points  =   [ (offset_x, img.shape[0]),
                   ( img.shape[1]-offset_x, img.shape[0]) ,
                   ( img.shape[1]-offset_x, 5 ),
                   ( offset_x, 5 )  ]

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
