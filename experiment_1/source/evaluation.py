import tensorflow as tf
import matplotlib.pyplot as plt

from source.GAN import GAN
from source.data_loader import get_data



if __name__ == '__main__':
    path = 'training_checkpoints/ckpt-3_temp'

    train_ds, val_ds = get_data()

    gan = GAN()
    gan.checkpoint.restore(tf.train.latest_checkpoint(path))

    for input_image, real_image in val_ds.take(10):
        input_image = tf.expand_dims(input_image, axis=0)
        real_image = tf.expand_dims(real_image, axis=0)
        generated_image = gan.predict(input_image)

        print(input_image.shape)
        print(real_image.shape)
        print(generated_image.shape)

        plt.subplot(131)
        plt.imshow(input_image[0])
        plt.subplot(132)
        plt.imshow(real_image[0])
        plt.subplot(133)
        plt.imshow(generated_image[0])
        plt.show()
