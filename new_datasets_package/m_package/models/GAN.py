import keras
from keras.layers import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import Model


noise_shape = 128


def build_generator_ver1(): 
    model = keras.Sequential()
    
    model.add(Dense(5*15*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((5,15,128)))

    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation="relu", use_bias=False))
    
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation="relu", use_bias=False))
    
    model.add(Conv2DTranspose(128, (5, 5), strides=(3, 3), padding='same', activation="relu", use_bias=False))
    
    model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', activation="relu", use_bias=False))
    
    model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', activation="relu", use_bias=False))
    
    model.add(Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', activation="sigmoid", use_bias=False))

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


class ImageGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, save_dir, save_freq, ver):
        super(ImageGenerationCallback, self).__init__()
        self.generator = generator
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.ver = ver

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            batch_size = 16  
            noise = tf.random.normal([batch_size, 128])
            generated_images = self.generator(noise, training=False)

            for i in range(generated_images.shape[0]):
                 plt.subplot(4, 4, i+1)
                 plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                 plt.axis('off')
                 
            plt.savefig(f"{self.save_dir}/{self.ver}_generated_image_{epoch}.png")
            print(f"Generated images saved for epoch {epoch}")


class GANModel(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GANModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer, g_loss, d_loss, *args, **kwargs):

        super().compile(*args, **kwargs)

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.g_loss = g_loss
        self.d_loss = d_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.d_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.d_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.g_loss(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise_dim = self.generator.input_shape[1]
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {'g_loss': gen_loss, 'd_loss': disc_loss}
