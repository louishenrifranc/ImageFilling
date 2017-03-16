import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

import numpy as np

sys.path.append("..")
from data_augmentation import rotate


def read_and_decode(filename_queue,
                    batch_size):
    """
    Create a queue for the task of visualizing embedding
    Only return the caption and the true image
    :param filename_queue: All filenames
    :param batch_size: Size of the batch
    :return:
    """
    # Create a tfRecordReader
    reader = tf.TFRecordReader()

    # Read a single example
    _, image_file = reader.read(filename_queue)

    # All fixed length features
    context_features = {
        "img": tf.FixedLenFeature([], tf.string)
    }

    # For ease of use, I used sequential features even if I knew the length
    sequence_features = {}
    for index in range(5):
        sequence_features["caption{}".format(index)] = tf.VarLenFeature(dtype=tf.float32)

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=image_file,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Decode the raw float image
    image = tf.decode_raw(context_parsed["img"], tf.uint8)
    # Reshape the image. Here the number of channel varies between sample
    # Some images are 1D channel, some other are 3D
    image = tf.reshape(image, (64, 64, -1))
    image_unit32 = 2 * tf.image.convert_image_dtype(image, dtype=tf.float32) - 1

    # image_unit32 = tf.image.rgb_to_hsv(image)
    # If number of channel is 1  -> modify to rgb scale
    image_unit32 = tf.cond(pred=tf.equal(tf.shape(image_unit32)[2], 3),
                           fn1=lambda: image_unit32,
                           fn2=lambda: tf.image.grayscale_to_rgb(image_unit32))

    # Need to define the true shape
    image_unit32.set_shape((64, 64, 3))
    # image_unit32 = rotate(image_unit32, 30)

    min_queue_examples = 256  # Shuffle elements

    # Because I know the true shape, I don't need sparse tensor (introduced by  the use of sequence_features)
    # so i transform them to dense vector +  reshape them
    caption0 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption0"]), (4800, 1))
    caption1 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption1"]), (4800, 1))
    caption2 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption2"]), (4800, 1))
    caption3 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption3"]), (4800, 1))
    caption4 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption4"]), (4800, 1))

    inputs = [image_unit32,
              caption0,
              caption1,
              caption2,
              caption3,
              caption4,
              ]
    images = tf.train.batch(
        inputs,
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size)
    return images


if __name__ == '__main__':

    batch_size = 4
    writer_filename = [os.path.join("..", "examples", "train{}.tfrecords".format(i)) for i in range(3, 4)]
    filename_queue = tf.train.string_input_producer(
        writer_filename)
    images = read_and_decode(filename_queue, batch_size)
    with tf.Session() as sess:
        group = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess.run(group)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess)
    for i in range(1000):
        out = sess.run(images)

        for b in range(batch_size):
            ou = out[0][b]
            print(np.mean(ou))
            print(np.max(ou))
            print(np.min(ou))
