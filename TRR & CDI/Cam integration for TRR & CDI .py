# Script implements capturing of images through webcam on click of 'Space' key
# Image detection model classifies the image to 4 of the classses : 
    #1. Handwritten text , 2.Printed Text , 3. Paper Currency or 4. Metal coin currency 
# Based on detection text and curreny/coins value is converted to speech(audio)

#Note :Module is much faster with a GPU


#required imports
import os
import cv2
import numpy as np
from gtts import gTTS
from PIL import Image 
import easyocr
from playsound import playsound
from tensorflow.keras.models import load_model


#Drive links for .h5 model files
#1. Detection Model - https://drive.google.com/file/d/1low9yt8XfiFLuTgC_EVm9Ep44_OM6Yau/view?usp=sharing
#2. Paper currency - https://drive.google.com/file/d/1f22Ay09CbgVyJ1bPOoNlh3IaCmqbq8cI/view?usp=sharing
#3. Metal coins - https://drive.google.com/file/d/16ycM0mMX03-FJlZ5FS9Azylq4dsMJH-U/view?usp=sharing
#4. handwritten model pending



#define path variables for trained models(.h5 files)
model_path = "D:/ML projects/cam capture/DetectionModel_vgg_4classes.h5"
metal_coin_model_path = "D:/ML projects/COINS CLASSIFICATION TASK/models/metal_currency_classification.hdf5"
paper_model_path = "D:/ML projects/Paper Currency Classification/cnn_model.h5"


#define labels list
labels = ['Handwritten_Text','Metal_Currency','Paper_Currency','Printed_Text'] #Text and Coins classification labels
paper_labels = ['Ten Rupees','Hundred Rupees','Twenty Rupees','Two Hundred Rupees','Two Thousand Rupees','Fifty Rupees', 'Five Hundred Rupees','Background'] #paper currency classification labels
coin_labels = ['Head of a coin','One Rupee','Ten Rupee','Two Rupee','Five Rupee'] #metal coin classification labels



#load the models to memory 
detection_model = load_model(model_path) #Text and coins model
paper_model = load_model(paper_model_path) #paper currency model
coin_model = load_model(metal_coin_model_path) #coin classification model
reader = easyocr.Reader(['en']) #printed text recognition model


#start camera to capture images
def startCamera():
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("ProxVision CDI & TRR Detection Frame", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed will quit camera
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed will capture image
            img_name = "Image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
            #evaluate captured image
            detection(img_name)
            os.remove(img_name) #remove image
            
            img_counter += 1

    
    cam.release()    
    cv2.destroyAllWindows()


#choose the currecny and text models     
def choose_model(text,image_path):
    
    if text == 'Printed_Text':
        print("Printed text recognition in process...")
        recognize_printed_text(image_path)
    elif text == 'Paper_Currency':
        print("Paper currency classification in process...")
        recognize_paper_currency(image_path)
    elif text == 'Metal_Currency':
        print("Metal Currency classification in process...")
        recognize_metal_coins(image_path)
        
    #4. handwritten model pending
        


#printed text recognition
def recognize_printed_text(image_path):
    
    #recognize text using easyocr model loaded into memory 
    results = reader.readtext(image_path, detail = 0) 
    print("reading text......")

    if len(results)>0:
        for line in results:
            #print(line)
            text_to_speech(line)
    else :
        print("No text recognized")
    
    
    
#paper currency classification
def recognize_paper_currency(image_path):
    
    image = Image.open(image_path) #open image
    image = image.resize((500,250)) #reshape image input as per model requirements
    #image.show()
    image = np.asanyarray(image)
    
    #preprocess
    image = image.astype("float") / 255.0 
    x = np.expand_dims(image, axis=0)
    
    #prediction 
    prediction = paper_model.predict(x)
    
    #get predicted class
    text = paper_labels[np.argmax(prediction)]
    #print(text)
    text_to_speech(text)



    
#metal coin classification    
def recognize_metal_coins(image_path):
    
    image = Image.open(image_path) #open image
    image = image.resize((300,300)) #reshape image input as per model requirements
    #image.show()
    image = np.asanyarray(image)
    
    #preprocess
    image = image.astype("float") / 255.0 
    x = np.expand_dims(image, axis=0)
    
    #prediction 
    prediction = coin_model.predict(x)
    
    #get predicted class
    text = coin_labels[np.argmax(prediction)]
    #print(text)
    text_to_speech(text)     
    
    
#convert predicted text to audio
def text_to_speech(text):   
    
    file_path = "audio.mp3" #file path to save audio files
    speech = gTTS(text=text, lang='en', slow=False) #convert text to speech 
    speech.save(file_path) #save audio file
    playsound(file_path) #play sound
    
    #remove audio file
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")    


    
def detection(image_path):
    
    img=cv2.imread(image_path) #open image
    img=cv2.resize(img,(224,224)) #reshape image input as per model requirements
    img = np.reshape(img,[1,224,224,3])

    #prediction 
    prediction = detection_model.predict(img)
    
    #get predicted class
    text = labels[np.argmax(prediction)]
    print("Image has ",text)
    
    choose_model(text, image_path)
    



if __name__== "__main__" :
    startCamera()