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

def bgr2rgb( bgr_img ) :
    return cv2.cvtColor( bgr_img, cv2.COLOR_BGR2RGB )

def rgb2bgr( bgr_img ) :
    return cv2.cvtColor( bgr_img, cv2.COLOR_RGB2BGR )

def rgb2gray( rgb_img ) :
    return cv2.cvtColor( rgb_img, cv2.COLOR_RGB2GRAY )


def binary2rgb( bin_img ) :

    bin_u8 = np.uint8( bin_img * 255  )
    return cv2.merge( (bin_u8,bin_u8,bin_u8) )

def binary2gray( bin_img ) :

    return np.uint8( bin_img * 255, dtype=np.uint8  )


def show( img, title="", figsize=(12,7.2), cmap = None, ax=None ) :
    if ax is None :
        _, ax = plt.subplots(1,1, figsize=figsize )

    print( f"img.shape={img.shape} img.max()={img.max()}" )

    if len( img.shape ) == 2 :  # monochrome img
        cmap = cmap if cmap is not None else 'gray'
        if img.max() == 1:
            ax.imshow( img * 255, cmap=cmap  )
        else :
            ax.imshow( img, cmap=cmap )
    elif len( img.shape ) == 3 :  # rgb
        ax.imshow( img, cmap=cmap)

    ax.set_title( title )
    return ax


def show2( img1, img2, title1="", title2 = "", figsize=(16,5) ) :
    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=figsize )
    show( img1, title1, ax=ax1  )
    show( img2, title2, ax=ax2  )


def frames_generator( video_fname ) :
    """Returns a generator that yields succesive frames from a video
    in BGR format! """
    if not os.path.exists( video_fname ) :
        raise RuntimeError( "File not found: " + video_fname )

    cap = cv2.VideoCapture( video_fname )

    while cap.isOpened() :
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret :
            print( "Opened but no frame... Go figure...")
            return

        yield frame


    # When everything done, release the video capture object
    cap.release()


def enumerated_frames_gen( video_fname, start=0, end=None ) :
    """Returns a generator that yields succesive frames from a video
    in BGR format! """
    if not os.path.exists( video_fname ) :
        raise RuntimeError( "File not found: " + video_fname )

    cap = cv2.VideoCapture( video_fname )

    i = 0
    while cap.isOpened() :
        # Capture frame-by-frame
        if end is not None  and i >= end:
            cap.release()
            return #raise StopIteration()

        ret, frame = cap.read()

        if not ret :
            return #raise StopIteration( "No more frames?" )
            #raise RuntimeError( "Opened but no frame... Go figure...")

        if i >= start:
            yield (i - start, frame)

        i += 1

    # When everything done, release the video capture object
    cap.release()


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


def test()  :
    #%%
    for i in range( 0, None )  :
        print( i )

        if i > 10 :
            break
    #%%