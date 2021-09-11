###created by :Aswin R
### model type: combination of CNN and LSTM

import tensorflow as tf 
import tensorflow.keras 
from tensorflow.keras.layers import (Dense,Conv2D,
MaxPooling2D,Flatten,Dropout,BatchNormalization,Input)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, TimeDistributed
from configuration import act_funct,ker_init,pool,flt

import numpy as np

class model_v1:
    def __init__(self,
    activation,kernel_initializer,No_categories,filters,pool_size,

                        ):

        #self.name=name
        #self.location=location
        self.activation=activation
        
        #self.no_layers=no_layers
        #self.kernels=kernels
        self.filters=filters
        self.activation=activation
        
        self.kernel_initializer=kernel_initializer
        self.pool_size=pool_size
        #self.strides=strides
        self.No_categories=No_categories


    def build_model(self,activation,kernel_initializer,No_categories,filters,pool_size):
        #input shape must change according to data
        input_sh=Input(shape=(96,96,3))
        model=Conv2D(16,(self.filters),
        kernel_initializer=self.kernel_initializer,
        activation=self.activation)(input_sh)
        #model=BatchNormalization()(model)

        model=MaxPooling2D(pool_size=(self.pool_size))(model)
        
        model=Conv2D(32,(self.filters),
        kernel_initializer=self.kernel_initializer,
        activation=self.activation
        )(model)


        #model=BatchNormalization()(model)
        model=MaxPooling2D(pool_size=(self.pool_size)
        )(model)
        #model=BatchNormalization()(model)

        model=Conv2D(64,(self.filters),
        kernel_initializer=self.kernel_initializer,
        activation=self.activation
        )(model)
        #model=BatchNormalization()(model)
        model=MaxPooling2D(pool_size=(self.pool_size)
        )(model)
        #model=BatchNormalization()(model)

        #model=TimeDistributed(Flatten())(model)


        #model=LSTM(33,activation='tanh',recurrent_initializer='glorot_uniform')(model)
        #model=Dense(1024,activation='relu')(model)
        model=Dense(1024,activation='relu')(model)
        model=Dense(self.No_categories)(model)
        model = Model(inputs=input_sh, outputs= model)
        return model

        

get_model=model_v1(act_funct,ker_init,5,flt,pool)
mo=get_model.build_model(act_funct,ker_init,5,flt,pool)

mo.summary()