import cv2
import time
import random
# import winsound
import numpy as np
import speech_recognition as sr
from jetson import utils
from threading import Thread
from collections import deque

#from nav import smart_nav
from smart_search import smart_search

def open_onboard_camera():
    cam = utils.videoSource('csi://0')
    return cam

    # cap = cv2.VideoCapture(1)
    # return cap

# def on_key_release_wrapper(t, q: deque):
#     def on_key_release(key): #what to do on key-release
#         time_taken = round(time.time() - t, 2) #rounding the long decimal float
#         q.append(time_taken)
#         print("The key",key," is pressed for",time_taken,'seconds')
#         return False #stop detecting more key-releases
#     return on_key_release

# def on_key_press_wrapper(req_key):
#     def on_key_press(key): #what to do on key-press
#         if str(key) == req_key:
#             return False #stop detecting more key-presses
#     return on_key_press

# def get_key_press_length(req_key):
#     q = deque(maxlen=1)
#     on_key_press = on_key_press_wrapper(req_key=req_key)
#     with Listener(on_press = on_key_press) as press_listener: #setting code for listening key-press
#         press_listener.join()

#     t = time.time() #reading time in sec
#     on_key_release = on_key_release_wrapper(t=t, q=q)
#     with Listener(on_release = on_key_release) as release_listener: #setting code for listening key-release
#         release_listener.join()
#     key_press_length = q.pop()
#     return key_press_length

def main():
    cap = open_onboard_camera()
    smart_nav_pause = deque(maxlen=1)

    #smart_nav_thread = Thread(target=smart_nav, args=(cap, smart_nav_pause))
    #smart_nav_thread.daemon = False
    #smart_nav_thread.start()

    smart_search_thread = Thread(target=smart_search, args=(cap, smart_nav_pause))
    smart_search_thread.daemon = False
    smart_search_thread.start()

if __name__ == '__main__':
    main()
