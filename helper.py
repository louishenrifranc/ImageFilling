import os

import inputs
import tensorflow as tf


def create_queue(filename, batch_size):
    """
    Create the queue, and get a holder on the queue input
    :return: list of SparseTensorValue and Tensor
        The output of a queue
    """
    path = os.path.join(os.path.dirname(os.path.basename(__file__)), "examples", "{}.tfrecords".format(filename))
    filename_queue = tf.train.string_input_producer(
        [path])
    return inputs.read_and_decode(filename_queue, batch_size)


def _sample(mean, log_sigma):
    epsilon = tf.truncated_normal(tf.shape(mean))
    # emb_dim (100)
    return epsilon * tf.exp(log_sigma) + mean


def get_mask_recon(hiding_size=32, overlap_size=7):
    mask_recon = tf.pad(tf.ones([hiding_size - 2 * overlap_size, hiding_size - 2 * overlap_size]),
                        [[overlap_size, overlap_size], [overlap_size, overlap_size]])
    mask_recon = tf.reshape(mask_recon, [hiding_size, hiding_size, 1])
    mask_recon = tf.concat(2, [mask_recon] * 3)
    """
    ---------------
    |             |
    |     000     |
    |     000     |
    |             |
    ---------------
    """
    return 1 - mask_recon


def get_mask_hiding(hiding_size=32, image_size=64):
    pad_size = (image_size - hiding_size) / 2
    mask = tf.pad(tf.ones([hiding_size, hiding_size]), [[pad_size, pad_size], [pad_size, pad_size]])
    mask = tf.reshape(mask, [image_size, image_size, 1])
    mask = tf.concat(2, [mask] * 3)
    """
    ---------------
    |             |
    |     111     |
    |     111     |
    |             |
    ---------------
    """
    return mask



