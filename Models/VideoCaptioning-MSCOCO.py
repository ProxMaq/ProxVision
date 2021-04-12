import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import time
import pickle
from gtts import gTTS
from playsound import playsound

"""
Note First you have to downlaod the checkpoints to run the VideoCaptioning.py. Download them from here

Link:- https://drive.google.com/drive/folders/1s0jmNKmIHvjdiq39cvh2WEVKFjTUv_Id?usp=sharing

Download them and change the checkpoint path in the evaluate function.

"""


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):

    # forward_layer = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    # backward_layer = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', go_backwards=True)

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # self.lstm = tf.keras.layers.LSTM(self.units,
        #                                    return_sequences=True,
        #                                    return_state=True,
        #                                    recurrent_initializer='glorot_uniform')
        # forward_layer = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # backward_layer = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', go_backwards=True)

        # self.bidirectional = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)  # self.lstm(x) #self.bidirectional

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
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


with open('C:\\Users\Harsh\Desktop\Projects\ProxVision\Models\MS COCO/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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

    checkpoint_path = "C:\\Users\Harsh\Desktop\Projects\Models\Training-Checkpoints"

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
            image = cv2.imwrite('image'+'.jpg', resized_frame)
            description, attention_plot = evaluate("image.jpg")
            os.remove("image.jpg")

            description = " ".join(description)
            description = description.replace("<end>", "")
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
