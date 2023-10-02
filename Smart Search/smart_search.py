from collections import deque
import os
from tkinter import OUTSIDE
from xml.dom import NotFoundErr
import cv2
import time
import random
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import playsound
from jetson import utils
import config
import beepy as bp
import threading

freq = 2000
dur = 1500  # milliseconds

def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

def distance_finder(focal_object, width_in_frame, real_object_width):
    distance = (real_object_width * focal_object) / width_in_frame
    return distance

def imgRead(img, yolo, id1, obj_names):
    classes, scores, boxes = yolo.detect(img, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
    n = len(boxes)
    if n:
        x_r = [True if scores[i] > 0.1 else False for i in range(n) if classes[i] == id1]
        data_list = [[obj_names[classes[i]], boxes[i][2], boxes[i][:2]] for i in range(n) if classes[i] == id1]
        if len(x_r) > 0:
            return x_r, data_list
    return

stop_flag = False

def speak(text):
    tts = gTTS(text=text)
    filename = "voice_search.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# def beep_sound():
#     global stop_flag
#     while not stop_flag:
#         bp.beep(sound=1)
#         time.sleep(1.5)


def beep_sound(cap,model,id1,obj_names):

    cv2.destroyAllWindows()
    c=0
    while True:

        frame=cap.Capture()

        # print(ret)
        # print(frame)
        # cv2.imshow("frme",frame)
        # cv2.waitKey(2000)
        

        frame = utils.cudaToNumpy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outs = imgRead(frame, model, id1, obj_names)
        print(outs)
        if outs is not None:
            # c=0
            outs = list(zip(*outs))
            out = outs[0]
            if not out[0]:
                c+=1
                if c>=100:
                    break
            else:
                c=0
                bp.beep(sound=1)
                time.sleep(1.5)
        else:
            c+=1
            if c>=100:
                break
        

def smart_search(cap, smart_nav_pause: deque):
    global stop_flag
    beep_thread= None

    yoloNet = cv2.dnn.readNet(config.YOLO_TINY_WEIGHTS, config.YOLO_TINY_CONFIG)
    yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn.DetectionModel(yoloNet)
    model.setInputParams(size=(416, 416), scale=1. / 255., swapRB=True)
    listn = sr.Recognizer()
    with open("coco1.names", 'r') as f:
        classes = f.read().splitlines()
        obj_names = [i.split('-')[0] for i in classes]
        obj_widths = [int(i.split('-')[1]) for i in classes]

    init_time = time.time()
    while True:
        if 24 <= ((time.time() - init_time) % 120) <= 30:
            smart_nav_pause.append(True)
            try:
                try:
                    with sr.Microphone() as source:
                        print("listening....")
                        speak("Hi Mahesh, what can I help you find today?")
                        time.sleep(4)
                        listn.adjust_for_ambient_noise(source)
                        voice = listn.listen(source)
                        command = listn.recognize_google(voice)
                        # command="find bottle";
                        print(command)
                        if 'find' in command:
                            obj = command.replace('find ', '').lower()
                            speak("finding the " + obj)
                        else:
                            smart_nav_pause.append(False)
                            continue
                except sr.UnknownValueError:
                    speak('Could not understand what you said')
                    smart_nav_pause.append(False)
                    continue
                except Exception as e:
                    print(e)
                    smart_nav_pause.append(False)
                    continue
                try:
                    id1 = obj_names.index(obj)
                except ValueError:
                    print('Couldn\'t find the object in detectable list')
                    smart_nav_pause.append(False)
                    continue
                print(f'finding {obj}')
                object_width = obj_widths[id1] * 0.0833333

                ref_object = cv2.imread(f'ReferenceImages/{obj}.jpeg')
                try:
                    _, object_data = imgRead(ref_object, model, id1, obj_names)
                except TypeError:
                    smart_nav_pause.append(False)
                    raise Exception('Model couldn\'t find object in Reference Image')

                object_width_in_rf = object_data[0][1]
                focal_object = focal_length_finder(config.KNOWN_DISTANCE, object_width, object_width_in_rf)
                time1 = time.time()
                frame_count = 0
                fps = {}
                #beep_thread = threading.Thread(target=beep_sound)
                #beep_thread.start()
                while True:
                    print("start")
                    time2 = time.time()
                    frame = cap.Capture()

                    # ret,frame = cap.read()
                    # cv2.imshow("frme",frame)
                    # cv2.waitKey(2000)

                    frame_count += 1
                    frame = utils.cudaToNumpy(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    outs = imgRead(frame, model, id1, obj_names)
                    if outs is not None:
                        outs = list(zip(*outs))
                        out = outs[0]
                        if not out[0]:
                            speak(str(obj_names[id1]) + " detected")
                        else:
                            distance = distance_finder(focal_object, out[1][1], object_width)
                            speak(str(obj_names[id1]) + " was found " + str(round(distance, 2)) + " feet away")
                            print(str(obj_names[id1]) + " is " + str(round(distance, 2)) + " feet away")
                            print("Object width in reference image is ", object_width_in_rf)
                            print("Focal object is ", focal_object)
                            print("Distance is ", distance)
                            print("Object width in coco1names is ", object_width)
                            print("Width in frame is ", out[1][1])
                            beep_thread = threading.Thread(target=beep_sound,args=(cap,model, id1, obj_names))
                            beep_thread.start()
                            break

                    if int(round(time2 - time1)) % 20 == 0:
                        print('diff', abs(int(round(time2 - time1))), 's')
                        if int(round(time2 - time1)) // 20 > 3:
                            print(f'Couldn\'t find {id1} \n Breaking out')
                            speak("Sorry, the " + obj + " was not found. Please try again.")
                            break
                        speak("still searching for " + obj)
                    fps['fps_'] = frame_count / (time.time() - time1)
                    if 'fps' not in fps:
                        fps['fps'] = fps['fps_']
                    fps['fps'] = 0.8 * fps['fps'] + 0.2 * fps['fps_']
                    print('FPS:', fps['fps'])

                # print("left.....")
                smart_nav_pause.append(False)
                # print("middle.....")
                stop_flag=True
                beep_thread.join()
                # print("right.....")
                break
            except Exception as e:
                smart_nav_pause.append(False)
                raise e

