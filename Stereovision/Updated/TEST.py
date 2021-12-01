import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# Filtering
kernel = np.ones((3, 3), np.uint8)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((8 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0, 11):  # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    #print(i)
    t = str(i)
    ChessImaR = cv2.imread('Right\chessboard-R' + t + '.png', 0)  # Right side
    ChessImaL = cv2.imread('Left\chessboard-L' + t + '.png', 0)  # Left side
    plt.figure(figure=(5,5))
    plt.imshow(ChessImaR)
    time.sleep(10.0)
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (8, 6),None)  # Right
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (8, 6), None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cornersR = cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cornersL = cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1], None, None)
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                            (wR, hR), 1, (wR, hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1], None, None)
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))


# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                           imgpointsL,
                                                           imgpointsR,
                                                           mtxL,
                                                           distL,
                                                           mtxR,
                                                           distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

# StereoRectify function
rectify_scale = 0  # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale,
                                                  (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1],
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)


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

frameL=cv2.imread('Left/ImageL2.png')
frameR=cv2.imread('Right/ImageR2.png')
print(Left_Stereo_Map[0].shape)
print(Left_Stereo_Map[1].shape)
Left0=Left_Stereo_Map[0].reshape(Left_Stereo_Map[0].shape[0],-1)
Left1=Left_Stereo_Map[1].reshape(Left_Stereo_Map[1].shape[0],-1)
sizeL=Left_Stereo_Map[0].shape+ Left_Stereo_Map[1].shape
Right0=Right_Stereo_Map[0].reshape(Right_Stereo_Map[0].shape[0],-1)
Right1=Right_Stereo_Map[1].reshape(Right_Stereo_Map[1].shape[0],-1)
sizeR=Right_Stereo_Map[0].shape+ Right_Stereo_Map[1].shape
#print(Left_Stereo_Map[0][0])
np.savetxt('Left_Stereo0.txt', Left0)
np.savetxt('Left_Stereo1.txt', Left1)
np.savetxt("sizesL.txt",sizeL)


np.savetxt('Right_Stereo0.txt', Right0)
np.savetxt('Right_Stereo1.txt', Right1)
np.savetxt("sizesR.txt",sizeR)

Lft0=(np.loadtxt("Left_Stereo0.txt", dtype=int))
L1=np.loadtxt("Left_Stereo1.txt",dtype=np.int).astype(np.float32)
SzeL=np.loadtxt("sizesL.txt",dtype=int).T
Rght0=np.loadtxt("Right_Stereo0.txt",dtype=int)
R1=np.loadtxt("Right_Stereo1.txt",dtype=int).astype(np.float32)
SzeR=np.loadtxt("sizesR.txt",dtype=int).T
L0=Lft0.reshape(SzeL[0],SzeL[1],SzeL[2]).astype(np.float32)
#L1=Lft1.reshape(SzeL[3],SzeL[4])
print(type(L1),type(Left_Stereo_Map[1]))
print(sum(L1-Left_Stereo_Map[1]))
R0=Rght0.reshape(SzeR[0],SzeR[1],SzeR[2]).astype(np.float32)


# Rectify the images on rotation and alignement
Left_nice = cv2.remap(frameL, L0, L1, cv2.INTER_LINEAR)  # Rectify the image using the kalibration parameters founds during the initialisation
Right_nice = cv2.remap(frameR,R0, R1, cv2.INTER_LINEAR)

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
filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

# Show the result for the Depth_image
cv2.imshow('Filtered Color Depth', filt_Color)
cv2.waitKey(0)