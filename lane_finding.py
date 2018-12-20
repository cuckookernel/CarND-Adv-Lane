# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:49:04 2018

@author: mrestrepo
"""
#pylint: disable=C0326
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2

import util as u

def pipeline_on_video( in_video_path, out_video_path, params, start_frame, end_frame )  :
    """Highest level function: Applies the whole pipeline to
    each frame of a video and outputs a new video with "search bands" highlighted around the
    lane-lines as well as the estimated lane curvature and the vehicle's horizontal offset.
    """

    frame_generator = u.enumerated_frames_gen( in_video_path,
                                               start=start_frame, end=end_frame )
    out_fps = 25
    try :
        video_writer = cv2.VideoWriter( out_video_path,
                                        cv2.VideoWriter_fourcc('M','J','P','G'),
                                        out_fps,  # fps : frames per second
                                        params["out_size_wh"] )

        print( "Writing video to: " + out_video_path )

        pipeline_on_many( frame_generator, params, video_writer )
    finally :
        video_writer.release()



def pipeline_on_many( frame_generator,  params, video_writer = None )  :

    tm0 = time.clock()

    detector = StableLaneDetector()

    for frame_i, frame_in in frame_generator :
        detector.frame_i = frame_i
        if frame_i == 0 :
            params["y_range"] = np.arange(0, frame_in.shape[0] )

        frame_out = detector.proc_one_frame( frame_in, params  )

        if video_writer :
           #assert len(out_img.shape) == 3, f"out_img.shape={out_img.shape}"
           bgr_out =u.rgb2bgr( cv2.resize(frame_out, params["out_size_wh"]) )
           video_writer.write( bgr_out )
    tm1 = time.clock()

    print( f"\n{ detector.frame_i } frames written in {tm1 -tm0:.2f} seconds "
           f"( { detector.frame_i / (tm1 - tm0):.2f} fps ) ")


class StableLaneDetector() :
    def __init__( self ) :
        self.R_smooth = EMASmoothener(alpha=0.05)
        self.fits = None
        self.fitxs = None
        self.frame_i = 0
        #self.left_smooth = EMASmoothener(alpha=0.1)
        #self.right_smooth = EMASmoothener(alpha=0.1)

    def proc_one_frame( self, frame, params  ) :
        """Apply pipeline to one frame and write and output image with annotations"""

        out_img, fits, fitxs, err_msg = pipeline_on_bgr_img( frame, params,
                                                             fits=self.fits,
                                                             fitxs=self.fitxs,
                                                             draw="lane" )

        yvalue_pix = frame.shape[0]

        if self.fits and fits :
            self.update_fits_with_sanity( fits, fitxs, params, yvalue_pix)
        elif fits:
            self.fits, self.fitxs = fits, fitxs
        else :
            print( f"frame {self.frame_i} : error: {err_msg}")
            return out_img

        smoothened_curve = self.R_smooth.get()

        m_x = 3.7 / (  self.fitxs[1][-1] - self.fitxs[0][-1] )
        car_x_offset = m_x * get_offset_pix( self.fits, yvalue_pix, out_img.shape[1] / 2 )

        cv2.putText( out_img,  f"Frame: {self.frame_i} smoothened radius of curvature: " +
                     f"{smoothened_curve:.0f} m",
                     (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), thickness=2 )

        cv2.putText( out_img, f"car horiz. offset: {car_x_offset:.2f} m",
                     (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), thickness=2 )

        return out_img

    def update_fits_with_sanity( self, fits, fitxs, params, yvalue_pix) :
        #Update the fits cache here...
        # First sanity test L1 norm of lane hasn't changed too much
        # (following grader suggestion)
        threshold = params["sane_update_threshold"]
        tmp_fit_l, tmp_fitx_l, upd_l = sane_fit_update_L1( self.fits[0], self.fitxs[0],
                                                           fits[0], fitxs[0], threshold )
        tmp_fit_r, tmp_fitx_r, upd_r = sane_fit_update_L1( self.fits[1], self.fitxs[1],
                                                           fits[1], fitxs[1], threshold )

        self.R_smooth.update( calc_curvature( (tmp_fit_l, tmp_fit_r),
                                              (tmp_fitx_l, tmp_fitx_r),
                                              params["ym_per_pix"], yvalue_pix ) )

        if self.frame_i % 10 == 0 :
            print( f"frame: {self.frame_i}  upd_l = {upd_l} upd_r = {upd_r}" )
# =============================================================================
#                    f"tmp_left_r_curve_m  = {tmp_left_r_curve_m:.2f} m "
#                    f"tmp_right_r_curve_m  = {tmp_right_r_curve_m:.2f} m "
#                    f"left_r_curve_m  = {left_r_curve_m:.2f} m "
#                    f"right_r_curve_m = {right_r_curve_m:.2f} m         ", end="\n" )
#
# =============================================================================

def calc_curvature( fits, fitxs, m_y, yvalue_pix ) :
    m_x = 3.7 / (  fitxs[1][-1] - fitxs[0][-1] )
    left_r_curve_m  = get_curvature_real_meters( fits[0], m_x, m_y, yvalue_pix )
    right_r_curve_m = get_curvature_real_meters( fits[1], m_x, m_y, yvalue_pix )
    r_curve = 0.5 * (left_r_curve_m + right_r_curve_m)

    return r_curve

#def curvatures_ok( fits, params, yvalue_pix ) :
#    m_y = params["ym_per_pix"]
#
#    m_x = 3.7 / 700 # value of mx doesn't matter to just check curvature ratio
#    left_r_curve_m  = get_curvature_real_meters( fits[0], m_x, m_y, yvalue_pix )
#    right_r_curve_m = get_curvature_real_meters( fits[1], m_x, m_y, yvalue_pix )
#    return 2.0 > left_r_curve_m / right_r_curve_m > 0.5


class EMASmoothener :
    """Exponential moving average smoothener """

    def __init__( self, alpha ) :
        """The smaller the alpha the slower it is"""
        self.value = None
        self.alpha = alpha

    def update( self, new_val) :
        """Update with fresh new value"""
        if self.value is None :
            self.value = new_val
        else :
            self.value = ( 1 - self.alpha ) * self.value + self.alpha * new_val

    def get( self ) :
        """Get smoothened value"""
        if self.value :
            return self.value
        else :
            return float("nan")


def calibrate_cam( img_filenames, n_x, n_y, draw=-1 ) :
    """Perform camera calibration via cv2.findChessboardCorners
    from a bunch of images stored in the paths contained in the list img_filenames.

    Return the camera matrix and the distortion coefficients
    """
    imgpoints = []
    objpoints = []

    objp = np.zeros( (n_x*n_y, 3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1,2)


    for i, fname in enumerate( img_filenames )  :
        img  = u.rgb_read( fname )
        #gray = u.rgb2gray( img )

        ret, corners = cv2.findChessboardCorners(u.rgb2gray( img ), (n_x,n_y), None)

        if draw == i :
            cv2.drawChessboardCorners(img, (n_x, n_y), corners, ret)
            plt.imshow( img )

        if ret :
            imgpoints.append( corners )
            objpoints.append( objp )

        print( f"img_filename = {fname} ret={ret} len(corners)="
               f"{len(corners) if corners is not None else 0} len(imgpoints)={len(imgpoints)}" )

    #perform calibration
    _, mtx, dist, _, _ = cv2.calibrateCamera( objpoints, imgpoints,
                                              (img.shape[1], img.shape[0]), None, None)

    return mtx, dist



def sane_fit_update_L1( sane_fit, sane_fitx, new_fit, new_fitx, threshold ) :
    l1_diff = np.average( np.abs(sane_fitx - new_fitx) )
    # print( f"l1_diff = {l1_diff:.2f}")
    test_ok = l1_diff <= threshold
    if test_ok :
        return new_fit, new_fitx, True
    else:
        return sane_fit, sane_fitx, False

    #return new_left_fitx, new_right_fitx

def get_offset_pix( fits, yvalue_pix, x_mid_pix ) :

    left_xvalue = np.polyval( fits[0], yvalue_pix )
    right_xvalue = np.polyval( fits[1], yvalue_pix )

    return x_mid_pix - 0.5 *( left_xvalue + right_xvalue )

def pipeline_on_bgr_img( frame, params, fits, fitxs, draw="search_windows"):

    mtx, dist, persp_m = ( params["cam_mtx"], params["cam_dist"],
                                        params["persp_M"], )


    warped_rgb, undistorted, lane_pixels = get_lane_pixels_bgr( frame, mtx, dist, persp_m )
    bin_gray = u.binary2gray( lane_pixels  )

    warp_bin = warp_perspective( bin_gray, persp_m )

    new_fits, new_fitxs, err_msg = detect_on_warp_bin( warp_bin, params, fitxs=fitxs )

    if new_fits : # detection ok
        fits  = new_fits
        fitxs = new_fitxs

    if fits :
        persp_m_inv = params["persp_M_inv"]

        if draw == "search_windows" :
            margin = params["margin"]
            img_rgb = draw_search_windows( warp_bin, fits[0], fits[1], margin, background = warped_rgb )
        elif draw == "lane" :
            img_rgb = draw_lane( warp_bin, fits[0], fits[1], background = warped_rgb )

        und1 = unwarp_combine( img_rgb, persp_m_inv, undistorted, inplace=True )

        return und1, fits,  fitxs, err_msg
        #return { "und1": und1, "warp_bin" :  warp_bin,
        #         "fits" : fits, "fitxs" : fitxs, "err_msg" : err_msg }
    else :
        return undistorted, fits, fitxs, err_msg
        #return { "und1": undistorted, "warp_bin" : warp_bin,
        #         "fits" : fits, "fitxs" : fitxs, "err_msg" : err_msg }


def detect_on_warp_bin( warp_bin, params, fitxs=None ) :
    """Apply the whole pipeline to a single image only
    return fits that satisfy curvature criterion"""

    window_width, window_height, margin = ( params["window_width"],
                                            params["window_height"],
                                            params["margin"] )

    if fitxs is None :
        # Hard work for frame 0...
        _ , left_lane, right_lane = sliding_window_search(warp_bin,
                                                          win_w=window_width,
                                                          win_h=window_height,
                                                          margin=margin)

        fits, fitxs = fit_polys0( left_lane, right_lane, params["y_range"] )

    else : # for frame # >= 1, use previous fit and function that search in a window around it
        left_fitx, right_fitx = fitxs
        fits, fitxs = fit_new_polys_v2( warp_bin, left_fitx, right_fitx,
                                        params["y_range"], margin=margin )

    #  Second Sanity Test (Following grader sugestion )
    m_y = params["ym_per_pix"]
    m_x = 3.7 / (fitxs[1][-1] - fitxs[0][-1])
    yvalue_pix = warp_bin.shape[0]

    left_r_curve_m  = get_curvature_real_meters( fits[0], m_x, m_y, yvalue_pix )
    right_r_curve_m = get_curvature_real_meters( fits[1], m_x, m_y, yvalue_pix )

    if not( 2.0 > left_r_curve_m / right_r_curve_m > 0.5 ) :
        err_msg = f"failed curvature test: {left_r_curve_m:.1f} {right_r_curve_m}  "
        return  None, None, err_msg  # lane finding failed curvature test
    else :
        return fits, fitxs, None


def get_lane_pixels_bgr( frame, mtx, dist, persp_m ) :
    """Convert from bgr to rgb, yndistort, perspective correct and get lane pixels"""

    distorted = u.bgr2rgb( frame )
    undistorted = cv2.undistort(distorted, mtx, dist, None, mtx)

    warped = warp_perspective(undistorted,  persp_m )

    return warped, undistorted, get_lane_pixels_0( undistorted )



def get_lane_pixels_0( rgb_correct ) :
    """Take a (perspective corrected) image and produce a binary image
      highlighting the lane pixels"""

    sobel_kernel=7

    hls = cv2.cvtColor(rgb_correct, cv2.COLOR_RGB2HLS)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    #gray = u.rgb2gray( orig )d
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
                  ( top_left_x, top_y ) ]

    return corners

def get_perspective_transform( img, corners, offset_x=100 ) :
    """get perspective transform"""
    dst_points  =   [ ( offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, 5 ),
                      ( offset_x, 5 ) ]

    dst = np.float32( dst_points )
    corners_nd = np.array( corners, dtype=np.float32 )
    persp_mat = cv2.getPerspectiveTransform( corners_nd, dst)

    return persp_mat, dst_points

def warp_perspective( und, persp_mat, flags=cv2.INTER_LINEAR) :
    """Apply perspective transform to an image"""

    return cv2.warpPerspective(und,  persp_mat, (und.shape[1], und.shape[0]),
                               flags=flags)


def draw_four_corners( imgc, corners, color = (255, 255, 0), tkn=3 ) :
    """Draw four segments  defined by four corner in a given order, using a
      color and thickness tkn """
    cv2.line( imgc, corners[0], corners[1], color, tkn )
    cv2.line( imgc, corners[1], corners[2], color, tkn )
    cv2.line( imgc, corners[2], corners[3], color, tkn )
    cv2.line( imgc, corners[3], corners[0], color, tkn )



def sliding_window_search(image, win_w, win_h, margin) :
    """ The code for this function borrows heavily from
    find_window_centroids from: Lesson 6 : Section7 - Another sliding window search.

    It just adds two return values ret_left, ret_right which are binary images
    with the same shapes as image and which will contain 1 in the pixels
    determined to be part of the left and right lanes, respectively
    """

    window_centroids = [] # Store the (left,right) window centroid positions per level
    ret_left  = np.zeros( image.shape, dtype=np.bool )
    ret_right = np.zeros( image.shape, dtype=np.bool )

    window = np.ones( win_w ) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to
    # get the vertical image slice and then np.convolve the vertical image slice
    # with the window template
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
        #convolve the window into the vertical slice of the image
        image_layer = np.sum(image[img_h - (level+1) * win_h :
                                   img_h - level * win_h, : ], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at
        # right side of window, not center of window

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
    #pylint: disable=R0913
    """For this level catch points in the left and right windows
    and place them in binary images ret_left, and ret_right"""

    img_h = warped.shape[0]

    slice_v  = slice( img_h - (level + 1) * win_h , img_h - level * win_h )

    slice_hl = slice( l_center - h_win_w, l_center + h_win_w )
    ret_left[ slice_v , slice_hl ] = ( warped[ slice_v, slice_hl ] > 0 )

    slice_hr = slice( r_center - h_win_w, r_center + h_win_w )
    ret_right[ slice_v , slice_hr ] = ( warped[ slice_v, slice_hr ] > 0 )



def fit_polys0( left_lane_img, right_lane_img, y_range ) :
    """Fit polinomials given separate left_lane and right_lane binary images """
    lefty, leftx   = left_lane_img.nonzero()
    righty, rightx = right_lane_img.nonzero()

    left_fit  = np.polyfit( lefty,  leftx,  deg=2 )
    right_fit = np.polyfit( righty, rightx, deg=2 )

    left_fitx  = np.polyval( left_fit,  y_range )
    right_fitx = np.polyval( right_fit, y_range )

    return (left_fit, right_fit), (left_fitx, right_fitx)


def fit_new_polys_v2(bin_warp, left_fitx, right_fitx, y_range, margin=100 ):
    """Fit new polynomials using left and right fit to draw window
    search stripes """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!

    # Grab activated pixels
    nonzeroy, nonzerox = bin_warp.nonzero()

    #yvals =  np.linspace(0, img_h - 1, img_h) # previously was nonzeroy....!

    #left_curve  = np.polyval( left_fit , nonzeroy )
    #right_curve = np.polyval( right_fit, nonzeroy )
    left_curve = left_fitx[ nonzeroy ]
    right_curve = right_fitx[ nonzeroy ]

    left_lane_inds  = ( (nonzerox < left_curve  + margin) &
                        ( nonzerox >= left_curve  - margin ) ).nonzero()[0]
    right_lane_inds = ( (nonzerox < right_curve + margin) &
                        ( nonzerox >= right_curve - margin ) ).nonzero()[0]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit  = np.polyfit( lefty,  leftx,  deg=2 )
    right_fit = np.polyfit( righty, rightx, deg=2 )

    left_fitx = np.polyval( left_fit, y_range )
    right_fitx = np.polyval( right_fit, y_range )

    return  (left_fit, right_fit), (left_fitx, right_fitx)


def draw_search_windows( bin_warp, left_fit, right_fit, margin,  background=None) :

    """ Visualization ##
     Create an image to draw on and an image to show the selection window
    """
    if background is None :
        background = u.binary2rgb( bin_warp )

    window_img = np.zeros_like(background)

    img_h = bin_warp.shape[0]

    ploty = np.linspace(0, img_h - 1, img_h )
     # left_fit[0] + left_fit[1] * ploty + left_fit[2] * ploty**2
    left_fitx  = np.polyval( left_fit, ploty )
    right_fitx = np.polyval( right_fit, ploty )

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud( np.vstack([left_fitx+margin, ploty]).T )])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([ np.vstack([right_fitx - margin, ploty]).T ])
    right_line_window2 = np.array([ np.flipud( np.vstack([right_fitx+margin, ploty]).T)])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    result = cv2.addWeighted(background, 1, window_img, 0.3, 0)

    return result


def draw_lane( bin_warp, left_fit, right_fit,  background=None) :

    """ Visualization ##
     Create an image to draw on and an image to show the selection window
    """
    if background is None :
        background = u.binary2rgb( bin_warp )

    window_img = np.zeros_like(background)

    img_h = bin_warp.shape[0]

    ploty = np.linspace(0, img_h - 1, img_h )

    left_fitx  = np.polyval( left_fit, ploty )
    right_fitx = np.polyval( right_fit, ploty )

    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line = np.array([ np.flipud( np.transpose(np.vstack([right_fitx, ploty]))) ])

    line_pts = np.hstack((left_line, right_line))

    cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

    result = cv2.addWeighted(background, 1, window_img, 0.3, 0)

    return result



def unwarp_combine( img0, mat_inv, undistorted, inplace=False ) :
    """
    Return a copy of undistorted pasting unwarped img0  on top of it.
    If inplace=True, the pasting will be done directly on undistored and
    undistored will be returned.
    """
    if inplace :
        und1 = undistorted
    else :
        und1 = undistorted.copy()

    unwarp = cv2.warpPerspective(img0, mat_inv,
                                 dsize= (und1.shape[1], und1.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    mask = (unwarp[...,0] != 0) & (unwarp[...,1] != 0) & (unwarp[...,2] != 0)

    und1[mask] = unwarp[mask]

    return und1


def get_curvature_real_meters( fit_in_pixel_space, mx, my, yvalue_pix ) :

    """Compute the a fit's radius  of curvature
    with the formula  in Lesson 7 : Section 7 but
    assuming the coefficients a, b,c that come in the fits are scaled according
    to mx = xm_per_pix  and my = ym_per_pix

    That is if, in pixel space:  x = a * y**2 + b * y + c
    and X = x *  mx , Y = y * my  are the corresponding coordinates in real space

    we have: X = mx/(my ** 2) * a * (Y**2) + (mx/my) * b * Y + (mx * c)

    This means that, in real
        X = A * (Y ** 2) + B * y + C

    with A := mx / (my ** 2) * a , and
         B := mx / my * b

    """
    #pylint: disable=C0103

    a , b, _ =  fit_in_pixel_space  # c is ignored...
    A , B  = ( mx / my ** 2 ) * a,  ( mx / my ) * b
    yvalue_m = yvalue_pix * my

    return ( 1 + (2 * A * yvalue_m + B)**2 ) ** 1.5 / abs( 2 * A )


def show_window_search_results( warped, window_centroids, window_w, window_h ) :
    """ The code for this function encapsulates the top level code
    from: Lesson 6 : Section7 - Another sliding window search
    """

    # If we found any window centers
    if window_centroids :

        # Points used to draw all the left and right windows
        l_points = np.zeros_like( warped )
        r_points = np.zeros_like( warped )

        # Go through each level and draw the windows
        for level, _  in enumerate(window_centroids) :
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_w, window_h, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_w, window_h, warped, window_centroids[level][1], level)
            #Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | (l_mask == 1) ] = 255
            r_points[(r_points == 255) | (r_mask == 1) ] = 255

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array( cv2.merge((zero_channel,template,zero_channel)),
                             dtype=np.uint8 ) # make window pixels green
        # making the original road pixels 3 color channels:
        warpage= np.dstack( (warped, warped, warped) ) # * 255
        # overlay the orignal road image with window results:
        #output = cv2.addWeighted( warpage, 1, template, 0.5, 0.0 )
        output = cv2.addWeighted( warpage, 0.7, template, 0.3, 0.0 )

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    return output

def window_mask(width, height, img_ref, center, level) :
    """ This function was taken from: Lesson 6 : Section7 - Another sliding window search
    """

    output = np.zeros_like( img_ref )
    output[int(img_ref.shape[0]-(level + 1) * height) : int(img_ref.shape[0]-level * height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output
