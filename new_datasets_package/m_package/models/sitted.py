import keras
from keras.layers import *


def sitted_deep_ConvLSTM1D():
    model = keras.Sequential()
    model.add(ConvLSTM1D(filters=32, 
                         kernel_size=7, 
                         activation='relu',
                         padding="same",
                         data_format='channels_last',
                         dropout=0.2, 
                         return_sequences=True,
                         input_shape=(10, 3, 1))) 
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM1D(filters=64, 
                         kernel_size=5, 
                         data_format='channels_last',
                         padding="same",
                         dropout=0.2, 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())

    model.add(ConvLSTM1D(filters=64, 
                         kernel_size=3, 
                         data_format='channels_last',
                         padding="same",
                         dropout=0.2, 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
    model.add(BatchNormalization())

    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))

    return model


def sitted_basic_ConvLSTM1D():
    model = keras.Sequential()
    model.add(ConvLSTM1D(filters=32, 
                         kernel_size=5, 
                         activation='relu',
                         padding="same",
                         data_format='channels_last',
                         dropout=0.2, 
                         return_sequences=True,
                         input_shape=(10, 3, 1))) 
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM1D(filters=64, 
                         kernel_size=3, 
                         data_format='channels_last',
                         padding="same",
                         dropout=0.2, 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
 
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))

    return model


def sitted_basic_ConvLSTM3D(shape):
        model = keras.Sequential()
        model.add(ConvLSTM2D(filters=32, 
                                 kernel_size=(5,5), 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True, 
                                 input_shape=(shape[0], shape[1], shape[2], 1))) 
        model.add(MaxPooling3D(pool_size=(1,1,2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, 
                                 kernel_size=(3,3), 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(2, activation="sigmoid"))
        return model



def sitted_deep_ConvLSTM3D(shape):
        model = keras.Sequential()
        model.add(ConvLSTM2D(filters=32, 
                                 kernel_size=(7,7), 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True, 
                                 input_shape=(shape[0], shape[1], shape[2], 1))) 
        model.add(MaxPooling3D(pool_size=(1,1,2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM2D(filters=64, 
                                 kernel_size=(5,5), 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM2D(filters=64,
                                 kernel_size=(3,3), 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation="sigmoid"))
        return model


