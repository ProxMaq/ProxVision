
"""
Python script for Indian Paper Currency classification 
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
model_path = "D:/ML projects/Paper Currency Classification/cnn_model.h5"
model = load_model(model_path)

#define labels 
labels = ['Ten Rupees','Hundred Rupees','Twenty Rupees','Two Hundred Rupees','Two Thousand Rupees',
          'Fifty Rupees', 'Five Hundred Rupees','Background']


#convert predicted text to audio
def text_to_speech(text):   
    
    file_path = "D:\ML projects\Paper Currency Classificationaudio.mp3" #file path to save audio files
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
    image = image.resize((500,250)) #reshape image input as per model requirements
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
    
    #image_url = "D:/ML projects/Paper Currency Classification/10_val_44.jpg"
    #image_url = "D:/ML projects/Paper Currency Classification/500_val_30.jpg"
    #image_url = "D:/ML projects/Paper Currency Classification/2000_val_35.jpg"
    #image_url = "D:/ML projects/Paper Currency Classification/50_val_0.jpg"
    image_url = "D:/ML projects/Paper Currency Classification/20_val_39.jpg"
    
    
    pred_text = coin_prediction(image_url)
    text_to_speech(pred_text)  