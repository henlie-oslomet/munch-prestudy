from __future__ import print_function, division
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization, Input, Dropout, Concatenate, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model

from source.utils import load_config

config = load_config()


def discriminator_layer(layer_input, filters, f_size=4, bn=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def conv2d(layer_input, filters, f_size=4, bn=True):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = Concatenate()([u, skip_input])
    return u


class Pix2Pix:
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss=config['discriminator_training']['loss'],
                                   optimizer=config['discriminator_training']['optimizer'],
                                   metrics=config['discriminator_training']['metrics'])

        target_image = Input(shape=config['image_shape'])
        input_image = Input(shape=config['image_shape'])

        output_image = self.generator(input_image)

        self.discriminator.trainable = False

        valid = self.discriminator([output_image, input_image])

        self.GAN = Model(inputs=[target_image, input_image], outputs=[valid, output_image])
        self.GAN.compile(loss=config['gan_training']['loss'],
                         loss_weights=config['gan_training']['loss_weights'],
                         optimizer=config['gan_training']['optimizer'])

    def build_generator(self):
        """U-Net Generator"""

        d0 = Input(shape=self.img_shape)

        filters = 64
        d1 = conv2d(d0, filters, bn=False)
        d2 = conv2d(d1, filters * 2)
        d3 = conv2d(d2, filters * 4)
        d4 = conv2d(d3, filters * 8)
        d5 = conv2d(d4, filters * 8)
        d6 = conv2d(d5, filters * 8)
        d7 = conv2d(d6, filters * 8)

        u1 = deconv2d(d7, d6, filters * 8)
        u2 = deconv2d(u1, d5, filters * 8)
        u3 = deconv2d(u2, d4, filters * 8)
        u4 = deconv2d(u3, d3, filters * 4)
        u5 = deconv2d(u4, d2, filters * 2)
        u6 = deconv2d(u5, d1, filters)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        filters = 64
        d1 = discriminator_layer(combined_imgs, filters, bn=False)
        d2 = discriminator_layer(d1, filters * 2)
        d3 = discriminator_layer(d2, filters * 4)
        d4 = discriminator_layer(d3, filters * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def get_gan(self):

        return self.GAN

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = np.add(d_loss_real, d_loss_fake) / 2

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      self.data_loader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
