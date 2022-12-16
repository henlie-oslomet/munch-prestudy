import os
import pathlib

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.image import imread
import tensorflow_addons as tfa
import math
import random

matplotlib.use('TkAgg')

from source.utils import load_config

tf.get_logger().setLevel('WARNING')

config = load_config()

directory = '../data/noise/usable/256_by_256'
masks = []
for filename in os.listdir(directory):
    mask = imread(os.path.join(directory, filename))
    mask = tf.Variable(mask, dtype='float32')
    mask = tf.reshape(mask, (config['image_height'], config['image_width'], 1))
    masks.append(mask)

upper = 90 * (math.pi / 180.0)
lower = 0 * (math.pi / 180.0)


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_png(image)

    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.grayscale_to_rgb(image)

    input_image = image
    real_image = image

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, config['image_height'], config['image_height'], 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def augment(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 256, 256)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    input_image = mask_batch(input_image)

    input_image, real_image = rotate(input_image, real_image)
    return input_image, real_image


def rotate(input_image, real_image):
    random_degree = random.uniform(lower, upper)
    input_image = tfa.image.rotate(input_image, random_degree)
    real_image = tfa.image.rotate(real_image, random_degree)
    return input_image, real_image


def mask_batch(image):
    # add random seed from config
    index = np.random.randint(0, len(masks))
    mask = masks[index]

    mask_value = 0.0 if config['augmentation']['testing'] else 255.0
    masked_image = tf.where(mask != 0.0, mask_value, image)
    return masked_image


def load_image(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = augment(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# def load_image_test(image_file):
#     input_image, real_image = load(image_file)
#
#     input_image, real_image = resize(input_image, real_image,
#                                      config['image_height'], config['image_width'])
#     input_image, real_image = normalize(input_image, real_image)
#
#     return input_image, real_image


def get_data():
    BUFFER_SIZE = 400

    train_path = '../data/SketchyDatabase/train/*/*.png'
    test_path = '../data/SketchyDatabase/test/*/*.png'
    train_path = pathlib.Path(train_path)
    test_path = pathlib.Path(test_path)

    train_dataset = tf.data.Dataset.list_files(str(train_path))
    test_dataset = tf.data.Dataset.list_files(str(test_path))

    train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(1)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_ds, _ = get_data()

    for input_image, real_image in train_ds.take(50):
        plt.subplot(121)
        plt.title('Input Image')
        plt.imshow(input_image[0] * 0.5 + 0.5)

        plt.subplot(122)
        plt.title('Real Image')
        plt.imshow(real_image[0] * 0.5 + 0.5)
        plt.show()
