# Package importation
import numpy as np
import cv2
import os

kernel = np.ones((3, 3), np.uint8)

Lft=np.loadtxt("Left_Stereo0.txt",dtype=np.int16)
Rt=np.loadtxt("Right_Stereo0.txt",dtype=np.int16)

L=Lft.reshape(Lft.shape[0],640,3)
L0=L[:,:,0:2]
L1=L[:,:,2]

R=Rt.reshape(Rt.shape[0],640,3)
R0=R[:,:,0:2]
R1=R[:,:,2]
# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 4
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


# StereoVision

frameL=cv2.imread('Left/ImageL1.png')
frameR=cv2.imread('Right/ImageR1.png')



# Rectify the images on rotation and alignement
Left_nice = cv2.remap(frameL, L0, L1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)  # Rectify the image using the kalibration parameters founds during the initialisation
Right_nice = cv2.remap(frameR,R0, R1, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

# Compute the 2 images for the Depth_image
disp = stereo.compute(grayL, grayR)
dispL = disp
dispR = stereoR.compute(grayR, grayL)
dispL = np.int16(dispL)
dispR = np.int16(dispR)

# Using the WLS filter
filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
# cv2.imshow('Disparity Map', filteredImg)
disp = ((disp.astype(
    np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect


# Filtering the Results with a closing filter
closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,
                            kernel)  # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)

# Colors map
dispc = (closing - closing.min()) * 255
dispC = dispc.astype(
    np.uint8)  # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)  # Change the Color of the Picture into an Ocean Color_Map
filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

# Show the result for the Depth_image
cv2.imshow('Filtered Color Depth', filt_Color)

# End the Programme
if cv2.waitKey(0) & 0xFF == ord(' '):
        cv2.destroyAllWindows()



