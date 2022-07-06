import speech_recognition as sr
import cv2
import numpy as np
import time
import datetime
import pyttsx3


listn = sr.Recognizer()

# download required weight and config file of the yolo model here https://pjreddie.com/darknet/yolo/
weight = r"yolov4-tiny.weights"
cfg = r"yolov4_tiny.cfg"
yolo = cv2.dnn.readNet(cfg, weight)

male_voice = pyttsx3.init()
#path of the system inbuilt voice from the system registry.
voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-GB_HAZEL_11.0"
male_voice.setProperty('voice', voice_id)

classes = []


def talk(text):
    male_voice.say(text)
    male_voice.runAndWait()

#coco name is the file which cointains all the 81 different object class on which the yolo model is trained.
with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

try:
    with sr.Microphone() as source:
        print("listening....")
        voice = listn.listen(source)
        command = listn.recognize_google(voice)

        if 'find' in command:
            command = command.replace('find ', '')
            talk("finding the " + command)
            id = classes.index(command)

except:
    pass

# id = classes.index('tv')
print(id)

def imgRead(img, x):

    num = x
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)

    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layer_name)

    boxs = []
    confidences = []
    class_ids = []

    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if id == class_id:
                if confidence > 0.05:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxs.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    x_r = True

    indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxs), 3))

    try:

        for i in indexes.flatten():
            x, y, w, h = boxs[i]
            label = str(classes[class_ids[i]])
            color = colors[i]

            # cv2.circle(img, (center_x, center_y), 2, (255,255,255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.imshow('img', img)

            if num != id:
                talk(str(classes[id]) + "detected")
            else:
                pass

        return x_r
    except:
        pass



if __name__ == "__main__":
    x = ''
    time1 = datetime.datetime.now()
    sec_flag = False
    while True:

        time2 = datetime.datetime.now()
        video = cv2.VideoCapture("https://192.168.0.2:8080/video")
        _, stream = video.read()
        cv2.imshow("frame", stream)
        out = imgRead(stream, x=x)
        video.release()

        if not out:
            imgRead(stream, x=x)

        else:
            x = id
            imgRead(stream, x=x)

        # and abs(int(round(time2.second - time1.second))) <= 34 and x != id
        if abs(int(round(time2.second - time1.second))) == 30:
            sec_flag = True
            talk("still searching for "+ str(classes[id]))

        if sec_flag:
            if abs(int(round(time2.second - time1.second))) == 0:
                talk("still searching for " + str(classes[id]))

        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
