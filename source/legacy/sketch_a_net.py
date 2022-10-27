import tensorflow as tf
from tensorflow.keras import layers

from source.data_loader import load_dataset
from source.utils import load_config

config = load_config()


def get_sketch_a_net_model():
    input_shape = config.get('image_height'), config.get('image_width'), 1

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(filters=64, kernel_size=(15, 15), strides=3, padding='valid', activation='relu'),  # L1
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),

        layers.Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='valid', activation='relu'),  # L2
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),

        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),  # L3

        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),  # L4

        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),  # L5
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'),

        layers.Conv2D(filters=512, kernel_size=(7, 7), strides=1, padding='valid', activation='relu'),  # L6
        layers.Dropout(rate=0.5),

        layers.Dense(units=512, activation='relu'),  # L7
        layers.Dropout(rate=0.5),

        layers.Dense(units=1000, activation='softmax'),  # L8
    ])
    model.build(input_shape=input_shape)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_ds, val_ds = load_dataset()
    model = get_sketch_a_net_model()
    model.fit(train_ds, epochs=100, callbacks=[], validation_data=val_ds)

