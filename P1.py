#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#AGR: short fix
rootDir = '../CarND-LaneLines-P1/'
roiY=330
#reading in an image
image = mpimg.imread(rootDir+'test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def calc_slope(line) :
    for x1,y1,x2,y2 in line:
        # to avoid NaN, we check the following and return a large value 
        # 1000.0 is practically vertical for most image resolutions
        #
        if(x1 == x2) :
            return 1000.0
        return ((y2-y1)/(x2-x1)) 
    
def x4y(line, ny, slope) :
   # slope = calc_slope(line)
    x1,y1 = line[0][:2]
    r =  (ny+slope*x1-y1)/slope
    if(np.isnan(r)) :
        print("Nan ")
        print(line)
        print(ny)
        print(slope)
    return r

def smooth(lines) :
    xs = np.array([l[0][0] for l in lines])
    A = np.array([ xs, np.ones(xs.size)])
    y = np.array([l[0][1] for l in lines])
    w = np.linalg.lstsq(A.T,y)
#    print("!!!")
#    print(w)
#    print(w[0])
    return w[0]
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    # verrtical lines will cause NPE in slope calculation. Filter out
    # could be avoided by conversion to polar coords, but not done for now
 #   print(lines)
    lines = [ l for l in lines if  np.abs(calc_slope(l))>0.5]
    if(len(lines)== 0) :
        return
    
    slopes = [calc_slope(l) for l in lines]
  #  print ("###################")
#    print ("---------------------------")
#    print (len(lines))
    a_slope = np.median(slopes)
    if(np.isnan(a_slope)) :
        print(slopes)
   # a_slope = smooth(lines)[0]
    bottomX = [x4y(l,img.shape[0],a_slope ) for l in lines]
#    print (bottomX)
#    print ("bot --" +str(a_slope))
#    print (np.mean(bottomX))
    a_bottomX = int(np.mean(bottomX))
    topX = [x4y(l,roiY, a_slope ) for l in lines]
#    print (topX)
#    print ("top --")
    a_topX = int(np.mean(topX))
#    print(a_topX)
#    print (img.shape)
    cv2.line(img, (a_topX ,roiY), (a_bottomX, img.shape[0]), color, 6)
#    print (calc_slope([[a_topX,roiY,a_bottomX,img.shape[0]]]))
#    print
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [0,255,0], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    h,w = img.shape
   # print("hough!!")
   # print(w)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
  #  print ("................. "+str(lines.size))
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, [l for l in lines if calc_slope(l) > 0 and l[0][0] > w/2])
    draw_lines(line_img, [l for l in lines if calc_slope(l) < 0 and l[0][0] < w/2])
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    imshape=image.shape
    # IMPROVE: calc dimensions based on real image size
    vertices = np.array([[(0,imshape[0]),(450, roiY), (490, roiY), (imshape[1],imshape[0])]], dtype=np.int32)
    result = grayscale(image)
    result = gaussian_blur(result,5)
    # 70, 150
    result = canny(result,70,150)
  #  mpimg.imsave('/home/andreas/NanoDegree/CarND-LaneLines-P1/test_images_output/canny-0.png',result)
    result = region_of_interest(result,vertices)
    # result = hough_lines(result,2,1,20,6,30)
    result = hough_lines(result,3,np.pi/180,30,6,30)
  #  mpimg.imsave('/home/andreas/NanoDegree/CarND-LaneLines-P1/test_images_output/hough-0.png',result)
    result = weighted_img(result,image)
    return result

import os
imgfiles = os.listdir(rootDir+"test_images/")
print(imgfiles)

#for imgfile in imgfiles:
#    print(imgfile)
#    image =  mpimg.imread(rootDir+'test_images/'+imgfile)
#    result = process_image(image)
#    plt.imshow(result,  cmap='gray')

rArray = [process_image(mpimg.imread(rootDir+'test_images/'+x)) for x in imgfiles]
# print (rArray)'
for index,o in enumerate(rArray) :
    print(cv2.imwrite('/home/andreas/NanoDegree/CarND-LaneLines-P1/test_images_output/out-'+str(index)+'.jpg',cv2.cvtColor(o, cv2.COLOR_RGB2BGR)))
    mpimg.imsave('/home/andreas/NanoDegree/CarND-LaneLines-P1/test_images_output/mp-'+str(index)+'.png',o)
    
print([plt.imshow(x,  cmap='gray') for x in rArray])

from moviepy.editor import VideoFileClip


white_output = rootDir+'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip(rootDir+"test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

y_output = rootDir+'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip(rootDir+"test_videos/solidYellowLeft.mp4")
y_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
y_clip.write_videofile(y_output, audio=False)

c_output = rootDir+'test_videos_output/challenge.mp4'
clip1 = VideoFileClip(rootDir+"test_videos/challenge.mp4")
c_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
c_clip.write_videofile(c_output, audio=False)
