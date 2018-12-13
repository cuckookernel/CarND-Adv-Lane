## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_chessboard.png "Undistorted"
[image2]: ./examples/undistorted_road_image.png "Road Transformed"
[image3]: ./examples/lanes_binary.png "Binary Example"
[image4]: ./examples/warped_road.png "Warp Example"
[image5]: ./examples/fit_after_window_search.png "Fit Visual"
[image6]: ./examples/road_image_w_lanes.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### IMPORTANT DEFINITIONS: 

Throughout this write-up we will use the following terms: 

  *  The **_Notebook_**:  refers to the file `Advanced Lane Finding.ipynb` which is the only notebook in the repository. 
  *  **All functions defined by me** and mentioned throughout are contained within `lane_finding.py` which is imported in the _Notebook_  as `p`. All line numbers refer to this single Python file also.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is implemented in the  `calibrate_cam` function (line 81). The way it was called to produce the camera 
and  distortion matrices is shown in the first code cell under header **Camera Calibration**  of the _Notebook_  ).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

In the _Notebook_, I applied this distortion correction to one of the calibration images (`./camera_cal/calibration3.jpg`) using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Applying distortion correction to given image is a simple matter of applying the 
`cv2.undistort` with the `mtx` and `dist` parameters computed in the previous step. 
This is done in the _Notebook_ on an image the `test_images/test6.jpg` to give the following 
result.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To construct a thresholded binary image I use the `get_lane_pixels0` (line 208) function which in turn uses a combination of thresholds for the following:

   * **scaled_x** The scaled (normalized) version of the _x_ component of the sobel transform of the lightness channel of the (HLS transformed) undistorted image. 
   * **mag_scaled** the normalized version of the magnitude (L2-norm) of the sobel 
   transform of the same. 
   * **s_channel** the saturation channel of the (HLS transformed) image. 
 
This particular values of the thresholds and the way to combine them come from a
decision tree that I calibrated with classical ML techniques from the example image and the corresponding binary version given in the lecture materials. 

Here is the the result of this method on the same road image used for the previous example:
   
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is divided into two parts.
   * Function `construct_trapezoid` (line 238) computes the locations of the four corners of an isoceles trapezoid whose base is a subsegment of the bottom of the image. The function includes parameters that determine what fraction of the base to include and how big the top side should be relative to the total with, and also how high should the top be.  After some fiddling with these parameters, I was able to align the sides of the trapezoid with the lines shown in  `straight_lines1.jpg`.
   
   * Function `get_perspective_transform` (line 265) computes the transform that maps the four corners of said trapezoid to a rectangle in the destination image which is centered. 
   
   The destination points are computed at the beginning of this function as :
```python
    dst_points  =   [ ( offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, img.shape[0]),
                      ( img.shape[1]-offset_x, 5 ),
                      ( offset_x, 5 )  ]
```
where `offset_x` is a parameter to the function that determines the left and right margins
of the rectangle.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane-line pixels in the warped binary image, I used the _"Sliding Window Search"_ given in the class materials with minor code cleanups. This method is implemented in the `sliding_window_search` (line 295) function. 

The fit to a quadratic polynomial is done in two different functions 

   * **fit_polys0**  : takes a left_lane and a right_lane binary image as returned from the first sliding window search identifies
   the non-zero points and the simply fits a polynomial to those points' coordinates
   
   * **fit_new_polys** : bypasses the sliding window search but takes advantage of previously computed fits. It takes a single binary
   image (which is simply the binarized warped road image) and uses the fits to find new lane points using stripes that have 
   width `2 * margin` and are centered around each fit. 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The real radius of curvature (in meters) is computed in function `get_curvature_real_meters` (line 483). This uses the basic curvature formula from 
lesson 7, section 7, after transforming the fit coefficients`a, b` from pixel space to their corresponding real space versions `A`, `B`.
The transformation is done following the observation. See the function's docstring for more details. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function `draw_search_windows` (line 416) takes a warped binary image of the road along with computed fits and margins 
draws search stripes around the lanes 
This function is called at the end of `pipeline_on_bgr_img` (line 159) which integrates all steps above in a single function.
After calling it we unwarp and fuse the result with original image . 
Here is an example: 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=UE4Oj_0wbXk&feature=youtu.be)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I had implementing this project, besides some struggle with the cv2 library API and error its messages, was 
to properly setup the combination of characteristics and thresholds used to produce the binary image in the `get_lane_pixels_0` (line 208) function. 
This involved a lot of trial and error and, when tried on the challenge and harder challenge videos, proved to be not very robust. 

As we see on those videos, the pipeline fails when there is a high gradient line that initially coincides with the side 
lane line but then separates. It can also easily fail if there is another vehicle crossing the lane in front of our car, when  there are weird light reflections that interfere with the capturing of the image, when the road ahead is not a horizontal plane and when the curves are very pronounced...

To make it more robust I would try several things: 
    - Rebalance the weight given to the x direction of the gradient vs. that given to the saturation channel. 
    - Apply some type of brightness adjustment preprocessing step before doing anything else with the image. 
    - Give a  greater weight to the lane points closer to our vehicle relative to those further away
    - Applying some kind of smoothing (or "filtering", e.g. Kalman filter) to the successive fits, to produce more stable fits and avoid that temporaray `distractions` mess with the fit. 


