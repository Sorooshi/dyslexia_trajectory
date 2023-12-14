import keras
from keras.layers import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import Model

noise_shape = 128

#generator model
def build_generator_lstm(image_shape, dense_image_shape):
        model = keras.Sequential()
        #noise processing
        model.add(InputLayer(input_shape=noise_shape))
        model.add(Dense(4 * 4 * 10 * 1, activation="relu")) 
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4,4,10,1)))

        #first decomposition block
        model.add(UpSampling3D())
        model.add(ConvLSTM2D(128, kernel_size=3, padding="same", data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))
        
        #second decomposition block
        model.add(UpSampling3D())
        model.add(ConvLSTM2D(64, kernel_size=3, padding="same", data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))

        #third decomposition block
        model.add(UpSampling3D())
        model.add(ConvLSTM2D(64, kernel_size=2, padding="same", data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))


        #out of the model
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(dense_image_shape, activation='tanh', name='Dense_Output')) 
        model.add(Reshape(image_shape))  
        return model
    
    
# discriminator model
def build_discriminator_lstm(image_shape):
        model = keras.Sequential()
        #first block
        model.add(ConvLSTM2D(64, kernel_size=5, input_shape=(image_shape[0], image_shape[1], image_shape[2], 1), padding="same", data_format='channels_last', return_sequences=True))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        #second block
        model.add(ConvLSTM2D(128, kernel_size=3, strides=2, padding="same", data_format='channels_last', return_sequences=True))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        #third block take this one for the expert layer weight
        model.add(ConvLSTM2D(128, kernel_size=3, strides=2, padding="same", data_format='channels_last', return_sequences=True))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

#CE model
def layer_block(inputs, num_filters, kernel_size):
    x = ConvLSTM2D(num_filters, kernel_size=kernel_size, padding="same", data_format='channels_last', return_sequences=True)(inputs)
    x = Activation("leaky_relu")(x)
    x = Dropout(0.4)(x)
    return x


def class_expert_model_lstm(weights, input_shape=(20,16,64,1), num_classes=2):
    inputs = Input(shape=input_shape)
    x = layer_block(inputs, 64, 5)
    x = layer_block(x, 128, 3)
    #freeze block class 1
    conv_non_trainable1 = ConvLSTM2D(256, kernel_size=3, padding="same", data_format='channels_last', name="freeze1", return_sequences=True)
    conv_non_trainable1.trainable = False
    x1 = conv_non_trainable1(x)
    x1 = Activation("leaky_relu")(x1)
    x1 = Dropout(0.4)(x1)
    #freeze block class 2
    conv_non_trainable2 = ConvLSTM2D(256, kernel_size=3, padding="same", data_format='channels_last', name="freeze2", return_sequences=True)
    conv_non_trainable2.trainable = False
    x2 = conv_non_trainable2(x)
    x2 = Activation("leaky_relu")(x2)
    x2 = Dropout(0.4)(x2)
    #final block
    x = Add()([x1, x2])
    x = Flatten()(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.layers[7].set_weights(weights[0])
    model.layers[7].set_weights(weights[1])
    return model
