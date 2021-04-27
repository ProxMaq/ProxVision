###created by :Aswin R
### model type: combination of CNN and LSTM

import tensorflow as tf 
import keras 
from tf.keras.layers import Dense,Conv2D,
MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential, Model
from keras.layers import LSTM, TimeDistributed
import numpy as np

class model_v1:
    def __init__(self,name,location,
    kernels,filters,activation,
    kernel_initializer,data_formats
    loss_funct,no_layers,
    pool_size,strides

                        ):

        self.name=name
        self.location=location
        self.activation=activation
        self.loss_funct=loss_funct
        self.no_layers=no_layers
        self.kernels=kernels
        self.filters=filters
        self.activation=activation
        self.loss_funct=loss_funct
        self.kernel_initializer=kernel_initializer
        self.data_formats=data_formats
        self.pool_size=pool_size
        self.strides=strides


    def build_model(self):
        
        model=Conv2D(self.kernels,(self.filters),
        kernel_intializer=self.kernel_initializer,
        data_formats=self.data_formats,
        activation=self.activation)
        model=BatchNormalization()(model)

        model=MaxPooling2D(pool_size=(self.pool_size),
        strides=(self.strides))(model)
        
        model=Conv2D()(model)
        model=MaxPooling2D()(model)
        model=BatchNormalization()(model)

        model=LSTM()
        



    def compile_model(self):



    def fit(self):
