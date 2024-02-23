import keras
from keras.layers import *
import numpy as np
import tensorflow as tf


def build_deep_conv3d(shape):
        model = keras.Sequential()
        model.add(Conv3D(filters=32, 
                     kernel_size=(7, 7, 7), 
                     activation='relu', 
                     data_format='channels_last',
                     input_shape=(shape[0], shape[1], shape[2], 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(1,1,2), padding='same', data_format='channels_last'))
        model.add(Conv3D(filters=64, 
                     kernel_size=(5, 5, 5), 
                     activation='relu', 
                     data_format='channels_last'))
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(Conv3D(filters=64, 
                     kernel_size=(3, 3, 3), 
                     activation='relu', 
                     data_format='channels_last')) 
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='sigmoid'))
        return model

def conv3d_model(shape):
    model = keras.Sequential()
    model.add(Conv3D(filters=32, 
                     kernel_size=(5, 5, 5), 
                     activation='relu', 
                     data_format='channels_last',
                     input_shape=(shape[0], shape[1], shape[2], 1))) 
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=64, 
                     kernel_size=(3, 3, 3), 
                     activation='relu', 
                     data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    return model


def build_deep_ConvLSTM1D(window_size):
        model = keras.Sequential()
        model.add(ConvLSTM1D(filters=32, 
                         kernel_size=5, 
                         activation='relu',
                         data_format='channels_last',
                         dropout=0.2, 
                         input_shape=(1 , window_size, 3), 
                         return_sequences=True)) 
        model.add(MaxPooling2D(pool_size=(1,2), padding="same", data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM1D(filters=64, 
                                 kernel_size=3, 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling2D(pool_size=(1,2), padding="same", data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM1D(filters=64,
                                 kernel_size=1, 
                                 activation="relu", 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling2D(pool_size=(1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation="sigmoid"))
        return model


def basic_ConvLSTN1D(window_size):
    model = keras.Sequential()
    model.add(ConvLSTM1D(filters=32, 
                         kernel_size=5, 
                         activation='relu',
                         data_format='channels_last',
                         dropout=0.2, 
                         input_shape=(1 , window_size, 3), 
                         return_sequences=True)) 
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=64, 
                         kernel_size=3, 
                         activation='relu', 
                         return_sequences=True)) 
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    return model


def conv1d_build_deep(window_size):
    model = keras.Sequential()
    model.add(Conv1D(filters=32, 
                     kernel_size=5, 
                     activation='relu', 
                     input_shape=(window_size, 3))) 

    model.add(BatchNormalization())

    model.add(Conv1D(filters=64, 
                     kernel_size=3, 
                     activation='relu')) 

    model.add(BatchNormalization())

    model.add(Conv1D(filters=64,
                     kernel_size=1, 
                     activation='relu')) 
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    return model

def conv1d_model(window_size):
    model = keras.Sequential()
    model.add(Conv1D(filters=32, 
                     kernel_size=5, 
                     activation='relu', 
                     input_shape=(window_size, 3))) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, 
                     kernel_size=3, 
                     activation='relu')) 
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    return model

def lstm_build_deep(window_size):
    model = keras.Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(window_size, 3))) 
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True)) 
    model.add(BatchNormalization())

    model.add(LSTM(64)) 
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    return model


def lstm_model_basic(window_size):
    model = keras.Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(window_size, 3)))
    model.add(BatchNormalization())
    model.add(LSTM(64)) 
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))
    return model


def conv2d_model(): #working done
        model = keras.Sequential()
        model.add(Conv2D(filters=32, 
                                 kernel_size=(5,5), 
                                 activation='relu', 
                                 input_shape=(60, 180, 1))) 
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, 
                                 kernel_size=(3,3), 
                                 activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(2, activation='sigmoid'))
        return model

def conv2d_build_deep():
    model = keras.Sequential()
    model.add(Conv2D(filters=32, 
                     kernel_size=(7,7), 
                     activation='relu', 
                     input_shape=(60, 180, 1))) 
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, 
                     kernel_size=(5,5), 
                     activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64,
                     kernel_size=(3,3), 
                     activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    return model


def build_basic(func, shape, classes, dense_activation):
        model = keras.Sequential()
        model.add(ConvLSTM2D(filters=32, 
                                 kernel_size=(5,5), 
                                 activation=func, 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True, 
                                 input_shape=(shape[0], shape[1], shape[2], 1))) 
        model.add(MaxPooling3D(pool_size=(1,1,2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=64, 
                                 kernel_size=(3,3), 
                                 activation=func, 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(classes, activation=dense_activation))
        return model


def build_deep(func, shape, classes, dense_activation):
        model = keras.Sequential()
        model.add(ConvLSTM2D(filters=32, 
                                 kernel_size=(7,7), 
                                 activation=func, 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True, 
                                 input_shape=(shape[0], shape[1], shape[2], 1))) 
        model.add(MaxPooling3D(pool_size=(1,1,2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM2D(filters=64, 
                                 kernel_size=(5,5), 
                                 activation=func, 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(ConvLSTM2D(filters=64,
                                 kernel_size=(3,3), 
                                 activation=func, 
                                 data_format='channels_last',
                                 dropout=0.2, 
                                 return_sequences=True)) 
        model.add(MaxPooling3D(pool_size=(1, 1, 2), padding='same', data_format='channels_last'))
        model.add(BatchNormalization())
    
        model.add(Flatten())
        model.add(Dense(128, activation=func))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation=func))
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation=dense_activation))
        return model







batch_size = 16


class ConvLSTM:
    def __init__(self, model, optimizer, loss_fn, train_metric, val_metric):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.loss_per_training = []
        self.training_auc = []
        self.valid_auc = []

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_metric.update_state(y, logits)
        return loss_value
    
    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        self.val_metric.update_state(y, val_logits)

    def fit(self, epochs, train_dataset, val_dataset):
        for epoch in range(epochs):
            train_losses = []
            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                train_losses.append(loss_value)
            train_auc = self.train_metric.result().numpy()
            print("Training AUC over an epoch: %.4f" % (float(train_auc),))
            print("Training average loss over an epoch: %.4f" % (float(np.nanmean(train_losses)),))

            self.loss_per_training.append(float(np.nanmean(train_losses)))
            self.training_auc.append(train_auc)
            
            self.train_metric.reset_states()

            for x_batch_val, y_batch_val in val_dataset:
                self.test_step(x_batch_val, y_batch_val)

            val_auc = self.val_metric.result().numpy()
            self.valid_auc.append(val_auc)
            self.val_metric.reset_states()
            print("Validation AUC: %.4f" % (float(val_auc),))


    def ret(self):
        return self.model