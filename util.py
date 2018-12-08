# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:51:50 2018

@author: mrestrepo
"""



import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path

greys_cmap = plt.get_cmap( 'Greys')


def rgb_read( path ) :
    if not os.path.exists( path ) :
        raise RuntimeError( "File not found: "  + path )

    maybe_img = cv2.imread( path )

    if maybe_img is None  :
        raise RuntimeError( "imread silently failed reading: "  + path +
                            "but file is there...")

    return maybe_img[:, :, ::-1].copy()

def rgb2gray( rgb_img ) :
    return cv2.cvtColor( rgb_img, cv2.COLOR_RGB2GRAY )


def binary2rgb( bin_img ) :

    bin_u8 = np.uint8( bin_img * 255  )
    return cv2.merge( (bin_u8,bin_u8,bin_u8) )


def show2( img1, img2, figsize=(16,5) ) :
    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=figsize )
    ax1.imshow( img1 )
    ax2.imshow( img2  )


def np_describe( np_arr, detail=0 )  :
    if detail == 0 :
        return { "shape" : np_arr.shape,
            "mean" : np_arr.mean(),
            "min" : np_arr.min(),
            "max" : np_arr.max(),
            "std" : np_arr.std(),
            "dtype" :  np_arr.dtype,

            }
    else :
        return { "shape" : np_arr.shape,
                "mean" : np_arr.mean(),
                "min" : np_arr.min(),
                "max" : np_arr.max(),
                "std" : np_arr.std(),
                "dtype" :  np_arr.dtype,
                "c_contiguous" : np_arr.flags.c_contiguous,
                "owndata" : np_arr.flags.owndata,
                "strides"  : np_arr.strides
         }
