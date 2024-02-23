import keras
from keras.layers import *
from keras.models import Model

def resnet_block(inputs, num_filters):
    x = Conv3D(num_filters, (1, 1, 1), padding='same', data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(num_filters, (3, 3, 3), padding='same', data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(num_filters * 4, (1, 1, 1), padding='same', data_format='channels_last')(x)
    x = BatchNormalization()(x)
    shortcut = inputs
    if inputs.shape[-1] != num_filters * 4:
        shortcut = Conv3D(num_filters * 4, (1, 1, 1), padding='same', data_format='channels_last')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def Resnet(input_shape=(20,16,64,1), num_classes=2):
    inputs = Input(shape=input_shape)
    #first block before resnet blocks
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same', data_format='channels_last')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2, 1, 2), padding='same')(x)

    # ResNet blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)

    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)

    #Dense blocks
    x = GlobalAveragePooling3D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_block_LSTM(inputs, num_filters):
    x = ConvLSTM2D(num_filters, (1, 1), padding='same', data_format='channels_last', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(num_filters, (3, 3), padding='same', data_format='channels_last', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ConvLSTM2D(num_filters * 4, (1, 1), padding='same', data_format='channels_last', return_sequences=True)(x)
    x = BatchNormalization()(x)
    shortcut = inputs
    if inputs.shape[-1] != num_filters * 4:
        shortcut = ConvLSTM2D(num_filters * 4, (1, 1), padding='same', data_format='channels_last', return_sequences=True)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def Resnet_LSTM(input_shape=(20,16,64,1), num_classes=2):
    inputs = Input(shape=input_shape)
    #first block before resnet blocks
    x = ConvLSTM2D(64, (7, 7), strides=(2, 2), padding='same', data_format='channels_last', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2, 1, 2), padding='same')(x)

    # ResNet blocks
    x = resnet_block_LSTM(x, 64)
    x = resnet_block_LSTM(x, 64)
    x = resnet_block_LSTM(x, 64)

    x = resnet_block_LSTM(x, 128)
    x = resnet_block_LSTM(x, 128)
    x = resnet_block_LSTM(x, 128)

    x = resnet_block_LSTM(x, 256)
    x = resnet_block_LSTM(x, 256)
    x = resnet_block_LSTM(x, 256)

    x = GlobalAveragePooling3D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_block_conv2d(inputs, num_filters):
    x = Conv2D(num_filters, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters * 4, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = inputs
    if inputs.shape[-1] != num_filters * 4:
        shortcut = Conv2D(num_filters * 4, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def Resnet_Conv2D(input_shape=(60,180,1), num_classes=2):
    inputs = Input(shape=input_shape)
    #first block before resnet blocks
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 1), padding='same')(x)

    # ResNet blocks
    x = resnet_block_conv2d(x, 64)
    x = resnet_block_conv2d(x, 64)
    x = resnet_block_conv2d(x, 64)

    x = resnet_block_conv2d(x, 128)
    x = resnet_block_conv2d(x, 128)
    x = resnet_block_conv2d(x, 128)
    x = resnet_block_conv2d(x, 128)

    x = resnet_block_conv2d(x, 256)
    x = resnet_block_conv2d(x, 256)
    x = resnet_block_conv2d(x, 256)
    x = resnet_block_conv2d(x, 256)
    x = resnet_block_conv2d(x, 256)
    x = resnet_block_conv2d(x, 256)

    x = resnet_block_conv2d(x, 512)
    x = resnet_block_conv2d(x, 512)
    x = resnet_block_conv2d(x, 512)

    #Dense blocks
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
