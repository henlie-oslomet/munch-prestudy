import os

import opensimplex
import tensorflow as tf
from utils import load_config
from itertools import product

config = load_config()
opensimplex.seed(config['random_seed'])


def generate_noise_image(x_start, x_end, y_start, y_end, limit, step):
    image = []

    for y in range(y_start, y_end):
        row = []
        y_step = y * step
        for x in range(x_start, x_end):
            x_step = x * step
            value = opensimplex.noise2(x_step, y_step)
            row.append(value)
        image.append(row)

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.where(image < limit, 0, 1)
    return image


def get_all_sub_files(folder):
    sub_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            sub_files.append(os.path.join(root, file))
    return sub_files


def extract_noise_in_range(low=0.1, high=0.2):
    number_of_pixels = config['image_height'] * config['image_width']
    low = number_of_pixels * low
    high = number_of_pixels * high

    dir_path = '../data/1/generated'
    files = get_all_sub_files(dir_path)
    from tqdm import tqdm
    for file in tqdm(files):
        noise = tf.io.read_file(file)
        noise = tf.image.decode_png(noise, channels=1)
        noise_sum = tf.reduce_sum(tf.cast(noise, tf.float32)) / 255
        if low < noise_sum < high:
            name = '_'.join(file.split('\\')[2:])
            tf.keras.utils.save_img(f'../data/1/usable/{config["image_height"]}_by_{config["image_width"]}/{name}', noise)


if __name__ == '__main__':
    # for x, y in product([0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.015, 0.02, 0.025]):
    #     folder = f'../data/1/generated/{y}/{x}'
    #     os.makedirs(folder, exist_ok=True)
    #     for i in range(0, 100):
    #         x_start = i * config['image_width']
    #         x_end = x_start + config['image_width']
    #         y_start = i * config['image_height']
    #         y_end = y_start + config['image_height']
    #
    #         image = generate_noise_image(x_start, x_end, y_start, y_end, x, y)
    #         image = tf.reshape(image, (config['image_height'], config['image_width'], 1))
    #         tf.keras.utils.save_img(f'{folder}/{i}.png', image)

    extract_noise_in_range()

