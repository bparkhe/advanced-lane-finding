
## Advanced Lane Finding Project
This project was executed as a par tof the Udacity Autonomous Vehicle Nano Degree Program. This project invoves a deeper understangding of the effect of color schemes on lane detection and estimating road trajectories which are required by the downstream planning and control algorithms.

The goals of this project are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undist.jpg "Undistorted"
[image2]: ./output_images/test_undist_and_warped.jpg "Undistorted and Warped Chessboard"
[image3]: ./test_images/test1.jpg "Test Image 1"
[image4]: ./output_images/test_undist.jpg "Undistorted Test 1 Image"
[image5]: ./output_images/test1_color_sobel_threshold.jpg "Color and Gradient Threshold Test 1"
[image6]: ./output_images/straight_lines1_mask.jpg "Straight Lines 1 with the mask for perspective transform"
[image7]: ./output_images/straight_lines1_warped.jpg "Straight Lines 1 Warped"
[image8]: ./output_images/test7.jpg "Custom test image"
[image9]: ./output_images/output_test1_polynomial.jpg "Polynomial identification using histogram"
[image10]: ./output_images/lane_output.jpg "Output test1 lane"
[video1]: ./output_images/project_video_output.mp4 "Output Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
[Rubric](https://review.udacity.com/#!/rubrics/571/view)

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

In this file I have addressed the requirements of the rubric. Read along!
The project execution pipeline for the video is in ./advanced_lane_finding.ipynb. The individual code blocks and step-by-step image output are in ./working_with_image.ipynb


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./advanced_lane_finding.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. (The functions might be commented out for speed.) I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


![alt text][image1]

An example of the warping function is also included for the chessboard images. The output of undistorted and warped image with the corners in the below result.

![alt text][image2]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to test image 1. Below is the original test 1 image. 
![alt text][image3]

To undistort this, I used the 'dist' and 'mtx' matrix from the camera calibration and passed it the cv2.undistort for this image. Below is the result. 
![alt text][image4]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I performed the thresholding after the warping step to give a binary output and maybe reduce the computational effort required. 
I converted the image to HLS and then applied thresholds of for the binary image 210 to 255 and sobel x gradient threshold 20 to 100.
Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


I used the below points to mark out a polygon on the image [255,650],[505,500],[802,500],[1100,650]
The polygon is plotted in the below image.
![alt text][image6]

I narrowed down to these coordinates by performing a perspective transform and the checking the outout for vertical lines.
I used the 'offset' value to control the extend the destination coordinates to extend the zone for identifying the lane.

'dst = np.float32([[offset, img_size[1]-offset],[offset, offset*8], [img_size[0]-offset, offset*8], 
                                 [img_size[0]-offset, img_size[1]-offset]])'

The perspective transform with the polygon can be seen below. Even though the polygon is mapped to a smaller set of area in the below  image, it doesnt affect the final output which would be similar if the polygon size was greater.
![alt text][image7]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To make sure the lane idetification works perfectly, I used an image from the video where there were more fringe gradients in the center of the image (shown below, shadows in the center of lane).

![alt text][image8]

The polynomial fitting to this image works by using the function find_lane_pixels
The flow for this process is as below:
1. Default lane coefficient set to [0,0,0] at the beginning of code block: pipeline
2. For the initial frame since the coefficients are unavailable, calculate them using fit_polynomial(function #3) 
    This is done using the if-else block in pipeline.
3. Fit polynomial uses finding_lane_pixels(fucntion #2) for histogram method  
4. The coefficients exist, the else block is activated. This uses the search_around_poly(function #5) with a margin of 50    around the previously detected polynomial
5. If any of the polynomial coefficients is 0, i.e when there are no lane pixels found, enter the if block again for histogram method

The output of the histogram method can be seen below.
There are many false pixels at the center of the image, but our algorithm sucessfully ignores them. 
![alt text][image9]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In the file advanced_lane_finding.ipynb, code block: Function.
I have defined function(#7) mesure_curvature_real that takes the x and y fitted polynomial values as its parameters. 
I have used the same values of xm_per_pix and ym_per_pix from the quiz for Measure Curvature 2. 
To get the polynomial coefficient values in meters, I recalculate the poynomial coefficients using np.polyfit with the existing polynomial x&y values converted to meters. This has been packaged in the generate_data function(#6).

Using the polynomial coeffcients, I calculate the radius of curvature using the formula. For the center offset, we are assuming that the camera is mounted exactly at the center of the dashboard. The calculated lane curve xcordinates at the bottom of the image are used (left_fitx(-1) and right_fitx(-1)). The average of these 2 values subtracted from the image center (640 pix) yields the offset value. Multiplying by xm_per_pix yileds the value in meters 

I printed out the values to the video using cv2.putText towards the end of code block: pipeline

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I unwarped the resulting image using the unwarp_to_original(function#8) and the inverse perspective transform Minv that was calculated between 'dst' and 'src'

Below is the output for the test1 image.

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4
![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The HSL and gradient thresholding values need further tuning to eliminate errors due to change in road texture. Current values are not that robust.

A spline can be used instead of 2nd order polynomial to get the lanes on the harder_challenge_example.

Smooting can be performed to reduce the wobbling of the lanes.
