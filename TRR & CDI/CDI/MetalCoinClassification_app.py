
"""
Python script for Indian Metal Coins classification 
@author: kalyani Avhale
"""

#imports
from keras.models import load_model
from playsound import playsound
from gtts import gTTS
from PIL import Image
import numpy as np 
import os


#load trained model
model_path = "D:/ML projects/COINS CLASSIFICATION TASK/models/metal_currency_classification.hdf5"
model = load_model(model_path)

#define labels 
labels = ['Head of coin','One Rupee','Ten Rupee','Two Rupee','Five Rupee']


#convert predicted text to audio
def text_to_speech(text):   
    
    file_path = "D:/ML projects/COINS CLASSIFICATION TASK/audio.mp3" #file path to save audio files
    speech = gTTS(text=text, lang='en', slow=False) #convert text to speech 
    speech.save(file_path) #save audio file
    playsound(file_path) #play sound
    
    #remove audio file
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("The file does not exist")
    
    
    
#perform prediction on images 
def coin_prediction(url):
    
    image = Image.open(url) #open image
    image = image.resize((300,300)) #reshape image input as per model requirements
    image.show()
    image = np.asanyarray(image)
    
    #preprocess
    image = image.astype("float") / 255.0 
    x = np.expand_dims(image, axis=0)
    
    #prediction 
    prediction = model.predict(x)
    
    #get predicted class
    text = labels[np.argmax(prediction)]
    #print(text)
    return text 

if __name__== "__main__" :
    
    image_url = "D:\ML projects\COINS CLASSIFICATION TASK/Test Images/Head_00001.JPG" #head
    #image_url = "D:\ML projects\COINS CLASSIFICATION TASK/Test Images/rupee1_000001.jpg" #rupee1
    #image_url = "D:\ML projects\COINS CLASSIFICATION TASK/Test Images/Rupee2_00001.jpg" #2rupee
    #image_url = "D:\ML projects\COINS CLASSIFICATION TASK/Test Images/Rupee5_0001.jpg"
    #image_url = "D:\ML projects\COINS CLASSIFICATION TASK/Test Images/Rupee10_00001.jpg"
    
    
    pred_text = coin_prediction(image_url)
    text_to_speech(pred_text)  
