# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Description of the pipeline

My pipeline consisted of the following steps:

* Calculation of ROI: Since some of the pics / videos have different format, the ROI area is not hardcoded but calculated based on the image width. (basically a polygon that is symmetric around the vertical middle of the imput image.

* The image is then converted to a gray image, to support gaussian blur and canny.
* Gaussian blur is applied
* Canny edge detection is applied to prepare for hough transformation
* The ROI mask is applied to the image (as drawn by the ROI caclulation)
* Lines are detected through Hough transformations. Parameters have been identified by manual tweaking.
* Lines are filtered based on their slope (nearly horizontal lines are discarded, e.g. the horizontal lines of lane markings).
* Lines are grouped based on the sign of their slope into left and right lane candidates.
* Slopes are averaged (using the median() function, which is a bit more stable wrt outliers than mean()).
* All endpoints of possible lines at the edges of the ROI are identified and averaged (again with median()):
* The resulting line are drawn in a given empty image
* The line image is superimposed onto the original image


### 2. Identify potential shortcomings with your current pipeline


* There is only support for detection of straight lines, not curved lines.
* Camera geometry (distortion) is not considered
* Low contrast or shadows can cause problems
* Car driving directly on a lane might cause additional problems
* Best suited for highway situations. Crossing line markings (such as on a city crossing) need extra logic
* Not suited for temporary lane markings on top of regular markings (such as yellow markings for temporary lines next to constructions sites).


### 3. Suggest possible improvements to your pipeline

* Take color into account (filter color based on HSL color space)
* Smooth by interpolating with previous frame lane detection
* Calibrate camera with camera geometry
* Use higher-order hough or spline interpolation


