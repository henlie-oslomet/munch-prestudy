#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: hed.py
# Author: Yuxin Wu

# import argparse
import numpy as np
import os
import shutil
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorpack import *
# from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
# from tensorpack.utils.gpu import get_num_gpu
# from tensorpack.utils import logger

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)

@layer_register(log_shape=True)
def CaffeBilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.

    Args:
        x (tf.Tensor): a NCHW tensor
        shape (int): the upsample factor

    Returns:
        tf.Tensor: a NCHW tensor.
    """
    inp_shape = x.shape.as_list()
    ch = inp_shape[1]
    assert ch == 1, "This layer only works for channel=1"
    # for a version that supports >1 channels, see:
    # https://github.com/tensorpack/tensorpack/issues/1040#issuecomment-452798180

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret

    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, 1, shape, shape], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, 1, shape, shape], 'SAME', data_format='NCHW')
    edge = shape * (shape - 1)
    deconv = deconv[:, :, edge:-edge, edge:-edge]

    if inp_shape[2]:
        inp_shape[2] *= shape
    if inp_shape[3]:
        inp_shape[3] *= shape
    deconv.set_shape(inp_shape)
    return deconv

class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, None, None, 3], tf.float32, 'image'),
                tf.TensorSpec([None, None, None], tf.int32, 'edgemap')]

    def build_graph(self, image, edgemap):
        image = image - tf.constant([104, 116, 122], dtype='float32')
        image = tf.transpose(image, [0, 3, 1, 2])
        edgemap = tf.expand_dims(edgemap, 3, name='edgemap4d')

        def branch(name, l, up):
            with tf.variable_scope(name):
                l = Conv2D('convfc', l, 1, kernel_size=1, activation=tf.identity,
                           use_bias=True,
                           kernel_initializer=tf.constant_initializer())
                while up != 1:
                    l = CaffeBilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up // 2
                return l

        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
            l = Conv2D('conv1_1', image, 64)
            l = Conv2D('conv1_2', l, 64)
            b1 = branch('branch1', l, 1)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            b2 = branch('branch2', l, 2)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            b3 = branch('branch3', l, 4)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            b4 = branch('branch4', l, 8)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            b5 = branch('branch5', l, 16)

            final_map = Conv2D('convfcweight',
                               tf.concat([b1, b2, b3, b4, b5], 1), 1, kernel_size=1,
                               kernel_initializer=tf.constant_initializer(0.2),
                               use_bias=False, activation=tf.identity)
        costs = []
        for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
            b = tf.transpose(b, [0, 2, 3, 1])
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = class_balanced_sigmoid_cross_entropy(
                b, edgemap,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)

        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, edgemap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        costs.append(wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        total_cost = tf.add_n(costs, name='cost')
        add_moving_summary(wrong, total_cost, *costs)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3e-5, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])



def run(model_path, image_path):
    pred = None
    try:
        pred_config = PredictConfig(
            model=Model(),
            session_init=SmartInit(model_path),
            input_names=['image'],
            output_names=['output' + str(k) for k in range(1, 7)])
        predictor = OfflinePredictor(pred_config)
        im = cv2.imread(image_path)
        assert im is not None
        print("im.shape", im.shape)
        # max_im_shape = max(im.shape)
        # if max_im_shape > 1000:
        #   scale_factor = 1000 / max_im_shape
        #   # print(scale_factor)

        #   width = int(im.shape[1] * scale_factor)
        #   height = int(im.shape[0] * scale_factor)
        #   dim = (width, height)

        #   im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)

        # im = cv2.resize(
        #     im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16)
        # )[None, :, :, :].astype('float32')

        im = cv2.resize(im, (256, 256))[None, :, :, :].astype('float32')

        print("im.shape", im.shape)

        outputs = predictor(im)
        pred = outputs[5][0]

        pred = np.uint8(pred[:,:,0] * 255)
        # Creating kernel
        kernel = np.ones((5, 5), np.uint8)

        # Using cv2.erode() method
        pred = cv2.erode(pred, kernel)
    except:
        print("Resize error.")


    return pred


model_filename = '../tensorpack/examples/HED/HED_pretrained_bsds.npz'
# image_filename = 'munch-sketches-final/multi-color/4707.jpg'

# image_res = run(model_filename, image_filename)
# # print(image_res[0,0,0])

# # print("image_res.shape", image_res.shape)

# edges = cv2.Canny(image=image_res, threshold1=100, threshold2=200) # Canny Edge Detection

# cv2.imshow('HED+Canny', edges)
# cv2.waitKey(0)

# cv2.destroyAllWindows()


# for path, subdirs, files in os.walk("munch-sketches-final"):
#     for name in subdirs:
#         subdir_name = os.path.join("edges-"+path, name)
#         print(subdir_name)
#         if os.path.exists(subdir_name):
#             shutil.rmtree(subdir_name)

#         os.makedirs(subdir_name)

#     for name in files:

#         input_filename = os.path.join(path, name)
#         output_filename = os.path.join("edges-"+path, name)
#         print(output_filename)

#         image_res = run(model_filename, input_filename)
#         if image_res is not None:
#             edges = cv2.Canny(image=image_res, threshold1=100, threshold2=200) # Canny Edge Detection
#             cv2.imwrite(output_filename, edges)


input_file_scream = os.path.join("munch-sketches-final", "multi-color", "7728.jpg")
output_file_hed = "hed_test.jpg"
output_file_hedcanny = "hedcanny_test.jpg"

image_res_scream = run(model_filename, input_file_scream)
if image_res_scream is not None:
    cv2.imwrite(output_file_hed, image_res_scream)
    edges_scream = cv2.Canny(image=image_res_scream, threshold1=100, threshold2=200) # Canny Edge Detection
    cv2.imwrite(output_file_hedcanny, edges_scream)
