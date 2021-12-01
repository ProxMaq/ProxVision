import cv2
import numpy as np
import os

vidR = cv2.VideoCapture(0)
#vidR.set(3,1280)
#vidR.set(4,720)
vidL = cv2.VideoCapture(2)
#vidL.set(3,1280)
#vidL.set(4,720)
directLeft="D:\WORK\python\Image Processing\Logitech\Left"
directRight="D:\WORK\python\Image Processing\Logitech\Right"

index_image=0
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while (True):
    retR, frameR = vidR.read()
    retL, frameL = vidL.read()
    flipRV=cv2.flip(frameR,-1)  # I have flippd this image because i have placed one camera with its head downward
    grayR= cv2.cvtColor(flipRV,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR,(8,6),None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
    retL, cornersL = cv2.findChessboardCorners(grayL,(8,6),None)  # Same with the left camera

    cv2.imshow('fliped',flipRV)
    cv2.imshow('frameL', frameL)

    if (retR == True) & (retL == True):
        corners2R= cv2.cornerSubPix(grayR,cornersR,(5,5),(-1,-1),criteria)    # Refining the Position
        corners2L= cv2.cornerSubPix(grayL,cornersL,(5,5),(-1,-1),criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(grayR,(8,6),corners2R,retR)
        cv2.drawChessboardCorners(grayL,(8,6),corners2L,retL)
        cv2.imshow('VideoR',grayR)
        cv2.imshow('VideoL',grayL)

        if cv2.waitKey(0) & 0xFF == ord('s'):   # Push "s" to save the images
            str_index_image= str(index_image)
            print('Images ' + str_index_image + ' saved for right and left cameras')
            os.chdir(directRight)
            cv2.imwrite('chessboard-R'+str_index_image+'.png',flipRV) # Save the image in the file where this Programm is located
            os.chdir(directLeft)
            cv2.imwrite('chessboard-L'+str_index_image+'.png',frameL)
            index_image=index_image+1
        else:                                # Push any key except "s" to not save the image
            print('Images not saved')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidR.release()
vidL.release()
cv2.destroyAllWindows()