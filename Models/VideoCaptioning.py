import time
import cv2
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from IPython.display import display

def startCamera(skip_seconds):

    video_path = "/some_path" # For local video captioning replace '0' from video_path
    cameraStream = cv2.VideoCapture(0)

    # Get frame information
    success, frame = cameraStream.read()
    height, width, layers = frame.shape
    #new_h = round(height / 2)
    #new_w = round(width / 2)
    new_h = 299
    new_w = 299

    # Get frames per seconds.
    fps = int(cameraStream.get(cv2.CAP_PROP_FPS))
    #CV_CAP_PROP_POS_FRAMES

    frame_count = 0

    skip_seconds
    # Empty string to fetch caption sentence output from model
    #caption_text = ''
    while (cameraStream.isOpened()):
        success, frame = cameraStream.read()
        if success == True:
            resized_frame = cv2.resize(frame, (new_w, new_h))
            #resized_frame = np.resize(frame, (new_w, new_h))

            # @Harsh uncomment below line and add your 'predict_caption()' method here.
            # caption_text = model.predict_caption(resized_frame)

            def extract_features(file, model):
                try:
                    image = file                
                except:
                    print("ERROR: Couldn't open image!")
                #image = image.resize((299,299))
                #image = np.array(image)
                # for images that has 4 channels, we convert them into 3 channels
                if image.shape[2] == 4:
                    image = image[..., :3]
                image = np.expand_dims(image, axis=0)
                image = image/127.5
                image = image - 1.0
                feature = model.predict(image)
                return feature

            def word_for_id(integer, tokenizer):
                for word, index in tokenizer.word_index.items():
                    if index == integer:
                        return word
                return None
            
            def generate_desc(model, tokenizer, photo, max_length):
                in_text = 'start'
                for i in range(max_length):
                    sequence = tokenizer.texts_to_sequences([in_text])[0]
                    sequence = pad_sequences([sequence], maxlen=max_length)
                    pred = model.predict([photo,sequence], verbose=0)
                    pred = np.argmax(pred)
                    word = word_for_id(pred, tokenizer)
                    if word is None:
                        break
                    in_text += ' ' + word
                    if word == 'end':
                        break
                return in_text

            
            #max_length = 32 # For Flickr8k
            max_length = 72 # For Flickr30k
            #tokenizer = load(open("Flickr8k/tokenizer.p","rb")) # For Flickr8k
            #model = load_model("Flickr8k/model_9.h5") # For Flickr8k
            tokenizer = load(open("Flickr30k/tokenizer.p","rb")) # For Flickr30k
            model = load_model("Flickr30k/model_8.h5") # For Flickr30k

            
            xception_model = Xception(include_top=False, pooling="avg")
            photo = extract_features(resized_frame, xception_model)
            #img = Image.open(img_path)

            description = generate_desc(model, tokenizer, photo, max_length)
            print(description)



            #print(f'{caption_text}') # later 'caption_text' will be converted into audio.

            # Commented, because our task is to generate caption (and later convert it into audio)
            cv2.imshow('frame', frame)

            if cv2.waitKey(fps) & 0xFF == ord('q'):
                print('Key \'Q\' is pressed !')
                break
        else:
            print('Video Capture returns FALSE !')
            break

        if skip_seconds > 0:
            frame_count += fps * skip_seconds
            cameraStream.set(1, frame_count)
            time.sleep(skip_seconds)

    # Release camera resource
    cameraStream.release()
    cv2.destroyAllWindows()
    print('Application exited !')

def runApp():
    print('Application started !')
    startCamera(3)

def main():
    runApp()
    
if __name__ == '__main__':
    main()