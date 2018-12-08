# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:57:04 2018

@author: mrestrepo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

import util  as u


def calibrate_cam( img_filenames, nx, ny, draw=-1 ) :
    """Perform camera calibration via cv2.findChessboardCorners
    from a bunch of images stored in the paths contained in the list img_filenames.

    Return the camera matrix and the distortion coefficients
    """
    imgpoints = []
    objpoints = []

    objp = np.zeros( (nx*ny, 3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)


    for i, fname in enumerate( img_filenames )  :
        img = u.rgb_read( fname )
        gray = u.rgb2gray( img )

        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if draw == i :
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            plt.imshow( img )

        if ret == True :
            imgpoints.append( corners )
            objpoints.append( objp )

        print( f"img_filename = {fname} ret={ret} len(corners)={len(corners) if corners is not None else 0}"
               f" len(imgpoints)={len(imgpoints)}" )


    #perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist