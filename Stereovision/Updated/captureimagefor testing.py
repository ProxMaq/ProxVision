import cv2
import numpy as np
import os
vidL = cv2.VideoCapture(2)
vidR = cv2.VideoCapture(0)
# vidL.set(3,1280)
# vidL.set(4,720)
# vidR.set(3,1280)
# vidR.set(4,720)
directLeft="D:\WORK\python\Image Processing\Logitech\Left"
directRight="D:\WORK\python\Image Processing\Logitech\Right"
# window_width=1280
# window_height=720
i=0


while (True):
    ret1, frameR = vidR.read()
    ret2, frameL = vidL.read()
    flipRV = cv2.flip(frameR, -1)
    cv2.imshow('frameRight',flipRV)
    cv2.imshow('frameLeft',frameL)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        i+=1
        fileL="ImageL"+str(i)+".png"
        fileR = "ImageR" + str(i) + ".png"
        os.chdir(directLeft)
        cv2.imwrite(fileL,frameL)
        os.chdir(directRight)
        cv2.imwrite(fileR, flipRV)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidR.release()
vidL.release()
cv2.destroyAllWindows()