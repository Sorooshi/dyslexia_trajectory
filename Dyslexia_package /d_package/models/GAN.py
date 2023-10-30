import numpy as np
import pandas as pd
import os
import cv2 as cv
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import tensorflow as tf 
from keras.layers import *
from keras.models import Sequential

latent_dim = 128
IMG_HEIGHT = 64
IMG_WIDTH = 128
SEQUENCE_LENGTH = 32

def generator_model_creation():
    model = Sequential()
    model.add(Input(shape=(latent_dim, )))
    model.add(Dense(4 * 8 * 16 * 128))
    model.add(Reshape((4, 8, 16, 128)))
    model.add(Conv3DTranspose(
        filters=128,
        kernel_size=4,
        strides=2,
        padding="same",
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3DTranspose(
        filters=256,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3DTranspose(
        filters=512,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3D(
        filters=3,
        kernel_size=5,
        padding="same",
        activation='sigmoid'
    ))

    model.summary()
    return model


def discriminator_model_creation():
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(Conv3D(
        filters=64,
        kernel_size=4,
        strides=2,
        padding='same',
        data_format='channels_last'
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3D(
        filters=128,
        kernel_size=4,
        strides=2,
        padding='same',
        data_format='channels_last'
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv3D(
        filters=256,
        kernel_size=4,
        strides=2,
        padding='same',
        data_format='channels_last'
    ))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, features):
        batch_size = tf.shape(features)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)

        combined_images = tf.concat([generated_images, features], axis=0)

        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
     
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }