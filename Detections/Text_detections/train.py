import keras
import tensorflow as tf 
import argparse
from model import mo
from configuration import epoch,optimizer,ls_funct
'''
train.py <path_to_train> <path_to validation>
'mo' i the model object called from model.py
'''

parser = argparse.ArgumentParser()
# get train data and validation data from cmd line
parser.add_argument('train_data', type=str,
                    help='path_to_train_dataset')
parser.add_argument('val_data', type=str,
                    help='path_to_validation_dataset')


args = parser.parse_args()

def main():
    print('[model is about to train.....]')
    #compile the model
    def compile_model():
        mo.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print('[model starts compiling.....]')
    compile_model()   
    from data import train_ds,val_ds
    #train the model
    def fit_model():
        print('[model starts training.....]')
        epochs = epoch
        history = mo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        )
    fit_model()

    tf.keras.models.save_model(
model=mo, filepath='/content/drive/MyDrive', overwrite=False, include_optimizer=True, save_format=tf,
signatures=None, options=None, save_traces=True
)
if __name__ == '__main__':
    main()