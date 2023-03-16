import numpy as np
import os
import cv2
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt



input_img = keras.Input(shape=(256, 256, 1))

# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# print(encoded.shape)

# # at this point the representation is (2, 2, 8) i.e. 128-dimensional

# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu')(x)
# x = layers.UpSampling2D((2, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# x = layers.Conv2D(16, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(input_img)
# x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(64, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(x)
# x = layers.Conv2D(128, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(256, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(512, kernel_size=3, strides=(2, 2), activation='leakyrelu', padding='same', kernel_initializer = 'he_normal', use_bias = False)(x)
# encoded = layers.BatchNormalization()(x)
# # print(encoded.shape)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(encoded)
# x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(x)
# x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(x)
# x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(x)
# x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(x)
# decoded = layers.Conv2DTranspose(1, kernel_size=3, strides=2, activation='leakyrelu', padding='same', use_bias = False)(x)
# print(decoded.shape)

def downsample(filters, size, apply_batch_normalization = True):
    downsample = keras.models.Sequential()
    downsample.add(keras.layers.Conv2D(filters = filters, kernel_size = size, strides = 2, use_bias = False, kernel_initializer = 'he_normal'))
    if apply_batch_normalization:
        downsample.add(keras.layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample

def upsample(filters, size, apply_dropout = False):
    upsample = keras.models.Sequential()
    upsample.add(keras.layers.Conv2DTranspose(filters = filters, kernel_size = size, strides = 2, use_bias = False, kernel_initializer = 'he_normal'))
    if apply_dropout:
        upsample.add(keras.layers.Dropout(0.1))
    upsample.add(keras.layers.LeakyReLU())
    return upsample


# encoder_input = keras.Input(shape = (SIZE, SIZE, 3))
x = downsample(16, 3, False)(input_img)
x = downsample(32,3)(x)
x = downsample(64,3,False)(x)
x = downsample(128,3)(x)
x = downsample(256,3)(x)

encoder_output = downsample(256,3)(x)
print("encoder_output.shape", encoder_output.shape)

decoder_input = upsample(256,3,True)(encoder_output)
x = upsample(256,3,False)(decoder_input)
x = upsample(128,3, True)(x)
x = upsample(64,3)(x)
x = upsample(32,3)(x)
x = upsample(16,3)(x)
x = keras.layers.Conv2DTranspose(8,(2,2),strides = (1,1), padding = 'valid')(x)
decoder_output = keras.layers.Conv2DTranspose(1,(2,2),strides = (1,1), padding = 'same')(x)


autoencoder = keras.Model(input_img, decoder_output)
#autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss='mean_absolute_error', metrics = ['acc'])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss='mean_squared_error', metrics = ['acc'])

dataset_list = []
for path, subdirs, files in os.walk("edges-munch-sketches-final"):
    for name in files:
        input_filename = os.path.join(path, name)
        im = cv2.imread(input_filename)
        im = cv2.resize(im, (256,256), interpolation = cv2.INTER_AREA)
        dataset_list.append(im[:,:,0])

dataset = np.array(dataset_list)
print("dataset.shape", dataset.shape)
train_idx = int(dataset.shape[0] * 0.9)

dataset_train = dataset[:train_idx]
dataset_test = dataset[train_idx:]

autoencoder.fit(dataset_train, dataset_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(dataset_test, dataset_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_imgs = autoencoder.predict(dataset_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(dataset_test[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(256, 256))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

encoder = keras.Model(input_img, encoder_output)
encoded_imgs = encoder.predict(dataset_test)

n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape((3*3, 256)).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

autoencoder.save('autoencoder')
encoder.save('encoder')
