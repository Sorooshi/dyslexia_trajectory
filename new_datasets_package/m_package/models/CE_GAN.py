import keras
from keras.layers import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from keras.models import Model

noise_shape = 128

# generator model
def build_generator(image_shape, dense_image_shape):
        model = keras.Sequential()
        #noise processing
        model.add(InputLayer(input_shape=noise_shape))
        model.add(Dense(4 * 4 * 10 * 1, activation="relu")) 
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4,4,10,1)))

        #first decomposition block
        model.add(UpSampling3D())
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last', name="1"))
        model.add(BatchNormalization(momentum=0.8))

        #second decomposition block
        model.add(UpSampling3D())
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last', name="2"))
        model.add(BatchNormalization(momentum=0.8))

        #third decomposition block
        model.add(UpSampling3D())
        model.add(Conv3D(64, kernel_size=(2, 2, 2), activation='relu', data_format='channels_last', name="3"))
        model.add(BatchNormalization(momentum=0.8))

        #out of the model
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))
        model.add(LeakyReLU(0.2))
        model.add(Dense(dense_image_shape, activation='tanh', name='Dense_Output')) 
        model.add(Reshape(image_shape))  
        return model

# discriminator model
def build_discriminator(image_shape):
        model = keras.Sequential()
        #first block
        model.add(Conv3D(64, kernel_size=5, input_shape=(image_shape[0], image_shape[1], image_shape[2], 1), padding="same", data_format='channels_last'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        #second block
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same", data_format='channels_last'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        #third block take this one for the expert layer weight
        model.add(Conv3D(256, kernel_size=3, strides=2, padding="same", data_format='channels_last'))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))
        
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

#training class (GAN)
class GAN(Model):
    def __init__(self, generator, discriminator, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size

        self.d_losses = []
        self.g_losses = []

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

        self.d_losses.append(total_d_loss.numpy())

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(tf.random.normal((self.batch_size,128,1)), training=True)

            predicted_labels = self.discriminator(gen_images, training=False)

            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        self.g_losses.append(total_g_loss.numpy())

        return {"d_loss":total_d_loss, "g_loss":total_g_loss}
    
    def plot_losses(self, name, path):
        fig, ax = plt.subplots( )
        tick = max(len(self.d_losses) // 5, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick))
        ax.plot(self.d_losses, label="Discriminator Loss")
        ax.plot(self.g_losses, label="Generator Loss")
        ax.legend()
        ax.title.set_text('model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        plt.savefig(f'{path}/{name}_loss.png', bbox_inches='tight') 



#CE model
def layer_block(inputs, num_filters, kernel_size):
    x = Conv3D(num_filters, kernel_size=kernel_size, padding="same", data_format='channels_last')(inputs)
    x = Activation("leaky_relu")(x)
    x = Dropout(0.4)(x)
    return x


def class_expert_model(weights, input_shape=(20,16,64,1), num_classes=2):
    inputs = Input(shape=input_shape)
    x = layer_block(inputs, 64, 5)
    x = layer_block(x, 128, 3)
    #freeze block class 1
    conv_non_trainable1 = Conv3D(256, kernel_size=3, padding="same", data_format='channels_last', name="freeze1")
    conv_non_trainable1.trainable = False
    x1 = conv_non_trainable1(x)
    x1 = Activation("leaky_relu")(x1)
    x1 = Dropout(0.4)(x1)
    #freeze block class 2
    conv_non_trainable2 = Conv3D(256, kernel_size=3, padding="same", data_format='channels_last', name="freeze2")
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
