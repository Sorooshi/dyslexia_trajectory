import keras
from keras.layers import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import Model

noise_shape = 128


def build_generator_ver1(dense_image_shape, size):
        model = keras.Sequential()
        #noise processing
        model.add(InputLayer(input_shape=noise_shape))
        model.add(Dense(4 * 10 * 1, activation="relu")) 
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4,10,1)))

        #first decomposition block
        model.add(UpSampling2D())
        model.add(Conv2D(128, 5, activation='relu', padding="same",data_format='channels_last', name="1"))
        model.add(BatchNormalization(momentum=0.8))

        #first decomposition block
        model.add(UpSampling2D())
        model.add(Conv2D(64, 5, activation='relu', padding="same", data_format='channels_last', name="2"))
        model.add(BatchNormalization(momentum=0.8))

        #first decomposition block
        model.add(UpSampling2D())
        model.add(Conv2D(64, 4, activation='relu', padding="same", data_format='channels_last', name="3"))
        model.add(BatchNormalization(momentum=0.8))

        #first decomposition block
        model.add(UpSampling2D())
        model.add(Conv2D(64, 2, activation='relu', padding="same", data_format='channels_last', name="4"))
        model.add(BatchNormalization(momentum=0.8))

        #out of the model
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(dense_image_shape, activation='tanh', name='Dense_Output')) 
        model.add(Reshape(size))
        return model


def build_generator_ver2(): 
    model = keras.Sequential()
    
    model.add(Dense(5*15*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((5,15,128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, (5, 5), padding='same', data_format='channels_last', activation="relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, (5, 5), padding='same', data_format='channels_last', activation="relu"))

    model.add(UpSampling2D(size=(3,3)))
    model.add(Conv2D(128, (5, 5), padding='same', data_format='channels_last', activation="relu"))

   
    model.add(Conv2D(128, (4, 4), padding='same', data_format='channels_last', activation="relu"))

    model.add(Conv2D(128, (4, 4), padding='same', data_format='channels_last', activation="relu"))

    model.add(Conv2D(1, (4, 4), padding='same', data_format='channels_last', activation="sigmoid"))

    return model



def build_discriminator(image_shape):
        model = keras.Sequential()
        model.add(Conv2D(32, kernel_size=5, input_shape=(image_shape[0], image_shape[1], 1), activation='relu' ,padding="same", data_format='channels_last'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu', padding="same", data_format='channels_last'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=3, strides=2, activation='relu', padding="same", data_format='channels_last'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", data_format='channels_last'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1))
        return model



class GAN(Model):
    def __init__(self, generator, discriminator, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):

        super().compile(*args, **kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        fake_images = self.generator(tf.random.normal((self.batch_size, 128, 1)), training=False)


        with tf.GradientTape() as d_tape:
            yhat_real = self.discriminator(real_images[0], training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((self.batch_size,128,1)), training=True)

            predicted_labels = self.discriminator(gen_images, training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss":total_d_loss, "g_loss":total_g_loss}
