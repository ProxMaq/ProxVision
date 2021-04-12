import tensorflow as tf
import cv2
import collections
import random
import re
import numpy as np
import os
import shutil
import time
import json
from glob import glob
from PIL import Image
from IPython.display import display
import pickle
from pickle import dump, load
from gtts import gTTS
from playsound import playsound


class Attention(tf.keras.Model):
    def __init(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def Call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = (tf.nn.tanh(
            self.W1(features)+self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()

        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(
            self.units, return_sequence=True, return_state=True, recurrent_initalizer='glorot_unitform')

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        x = self.fc1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def textToSpeech(text):
    speech = gTTS(text=text, lang='en', slow=False)
    speech.save(audio_file)
    playsound(audio_file)
    #---------For continous captioning----------#
    os.remove(audio_file)
    #-------------------------------------------#


with open('') as handle:
    tokenizer = load(handle)


def evaluate(image):

    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index) + 1
    attention_features_shape = 64  # To be filled after training on whole dataset
    max_length = 52  # 52 For the whole dataset

    attention_plot = np.zeros((max_length, attention_features_shape))

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    # Checkpoints
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "/content/drive/MyDrive/Image-Captioning/Training-Checkpoints"

    ckpt = tf.train.Checkpoint(
        encoder=encoder, decoder=decoder, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    # Prediction
    hidden = decoder.reset_state(batch_size=1)

    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def startCamera(skip_seconds):

    video_path = "/some_path"  # For local video captioning replace '0' from video_path
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
    # CV_CAP_PROP_POS_FRAMES

    frame_count = 0

    while (cameraStream.isOpened()):
        success, frame = cameraStream.read()
        if success == True:
            resized_frame = cv2.resize(frame, (new_w, new_h))

            # photo = extract_features(resized_frame, xception_model)
            #img = Image.open(img_path)

            # description = generate_desc(model, tokenizer, photo, max_length)
            description, attention_plot = evaluate(resized_frame)
            description = " ".join(description)
            print(description)

            # Convert text into speech
            textToSpeech(description)

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


audio_file = "caption_audio.mp3"
# -----------------------------------------------------


def runApp():
    print('Application started !')
    startCamera(3)


def main():
    runApp()


if __name__ == '__main__':
    main()


# image_path = '/content/datasets_122238_294885_Flickr_Data_Images_1009434119_febe49276a.jpg'

# display(Image.open(image_path))

# print('Predicted Caption:<start>', ' '.join(result))
