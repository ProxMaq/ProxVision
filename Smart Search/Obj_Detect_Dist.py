# import libraries
import cv2
import numpy as np
import datetime
import pyttsx3
import speech_recognition as sr

# pre-defined distance constants
KNOWN_DISTANCE = 45  # INCHES
OBJECT_WIDTH = 10  # INCHES


# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255),
          (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# function to calculate focal length 
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance*0.0833  # edit



# download required weight and config file of the yolo model here https://pjreddie.com/darknet/yolo/
weight = r"yolov4-tiny.weights"
cfg = r"yolov4-tiny.cfg"

# give the configuration and weight files for the model and load the network
yolo = cv2.dnn.readNet(cfg, weight)

# set the audio message config set up
male_voice = pyttsx3.init()
#path of the system inbuilt voice from the system registry.
voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-GB_HAZEL_11.0"
male_voice.setProperty('voice', voice_id)


# define the talk function for audio messages
def talk(text):
    male_voice.say(text)
    male_voice.runAndWait()

listn = sr.Recognizer()

# ask the user to find a particular object
try:
    with sr.Microphone() as source:
        print("listening....")
        talk("what can I help you find today?")
        voice = listn.listen(source)
        command = listn.recognize_google(voice)
              
        if 'find' in command:
            obj = command.replace('find ', '')
            talk("finding the " + obj)


except:
    pass

# obj = 'pottedplant'
print (obj)
#coco name is the file which cointains all the 81 different object class on which the yolo model is trained.
with open("coco.names", 'r') as f:
    classes = f.read().splitlines()
    
# find the id of the object to be detected    
id1 = classes.index(obj)
print(id1)

# reading the reference image (in .png format) from dir for calculating focal length
ref_object = cv2.imread(f'ReferenceImages/{obj}.png')

# object detection function
def imgRead(img, x):

    num = x
    height, width, _ = img.shape
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image = img, scalefactor = 1 / 255, size = (416, 416), mean = (0, 0, 0), swapRB=True, crop=False)
    # blob oject is given as input to the network
    yolo.setInput(blob)
    # get the index of the output layers
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    # forward pass
    layeroutput = yolo.forward(output_layer_name)

    # post processing
    boxs = []
    confidences = []
    class_ids = []
    data_list = []
    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if id1 == class_id:
                if confidence > 0.05:
                    
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # find corners 
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxs.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
                    x_r = True
    # get indexes of the object(s) detcted
    indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
    # assign font
    font = cv2.FONT_HERSHEY_PLAIN
    # assign colors to the bounding boxes
    colors = np.random.uniform(0, 255, size=(len(boxs), 3))
    
    # add bounding boxes to each object in the image frame
    try:
        for i in indexes.flatten():
            data_list = []
            x, y, w, h = boxs[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.imshow('img', img)

            if (num != id1) & (f==1):
                talk(str(classes[id1]) + "detected")
                data_list=[classes[class_ids[i]],w,(x,y)]

            else:
                data_list=[classes[class_ids[i]],w,(x,y)]
                pass
            
        return [x_r,data_list]
    except:
        pass

# detect the object in the reference image
_,object_data = imgRead(ref_object,obj)

# calculate the object width in real frame
object_width_in_rf = object_data[1]
# print('ppp',person_data)

# ensure that object dectection code for focal length calculation mutes the audio message
f = 0
# finding focal length of the camera
focal_object = focal_length_finder(KNOWN_DISTANCE, OBJECT_WIDTH, object_width_in_rf)
f = 1

# capture the webcam feed
cap = cv2.VideoCapture(0)
# pass blank index to the object detection function
x = ''
# counter to generate alert only once after the image is detected              
c= 0
# track time              
time1 = datetime.datetime.now()

while(True):
    # track time again to capture delay
    time2 = datetime.datetime.now()
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    out = imgRead(frame, x=x)

    if out!= None:

        if not out[0]:
                imgRead(frame, x=x)

        else:
                x = id1
                imgRead(frame, x=x)
                distance = distance_finder(focal_object, OBJECT_WIDTH, out[1][1])
                xl,yl = out[1][2]
                # draw bounding box and text with distance of the object
                cv2.rectangle(frame, (xl, yl-3), (xl+150, yl+23), BLACK, -1)
                cv2.putText(frame, f'Dis: {round(distance,2)} ft',
                   (xl+5, yl+13), FONTS, 0.48, GREEN, 2)
                cv2.imshow('frame', frame)
                # counter to generate audio with distance of the object only once
                c = c+1
                # once the object detected for the first time, audio message with the distance of the object from the camera
                if c == 1:
                    talk(str(classes[id1]) + "is" + str(round(distance,2)) + "feet away")

    if abs(int(round(time2.second - time1.second))) == 10:
        talk("still searching for "+ obj)       
    if cv2.waitKey(1) == ord('q'):
        break
        

cap.release()
cv2.destroyAllWindows()
