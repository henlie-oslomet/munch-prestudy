import numpy as np
import os
import cv2
# import tensorflow as tf
import keras
# from keras import layers
# from keras.callbacks import TensorBoard
# import matplotlib.pyplot as plt
import shutil


model = keras.models.load_model('encoder')

for path, subdirs, files in os.walk("munch-sketches-final"):
    for name in subdirs:
        subdir_name = os.path.join("compressed-"+path, name)
        print(subdir_name)
        if os.path.exists(subdir_name):
            shutil.rmtree(subdir_name)

        os.makedirs(subdir_name)

    for name in files:

        input_filename = os.path.join(path, name)
        output_filename = os.path.join("compressed-"+path, name)
        print(output_filename)
        im = cv2.imread(input_filename)
        im = cv2.resize(im, (256, 256))[None, :, :, 0].astype('float32')
        assert im is not None
        encoded_img = model.predict(im)

        print(encoded_img.shape)
        np.save(output_filename, encoded_img)


        # if encoded_img is not None:
        #     cv2.imwrite(output_filename, encoded_img)