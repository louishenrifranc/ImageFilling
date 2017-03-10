import os
import tensorflow as tf


def read_and_decode(filename_queue,
                    batch_size,
                    flip=True,
                    contrast=True):
    """
    Given a queue of filename, it reads every example stored
    in every TfRecord file, and accumulate them in batch
    Summary and Reviews are automatically padded with zeros
    :param filename_queue: A list of filename
    :return:
        A tuple containing a single batch
            (summary: batch_size x max_sequence_length_in_summary_batch,
            review: batch_size x max_sequence_length_in_review_batch,
            score: batch_size,
            reviewer_id: batch_size,
            film_id: batch_size)
            Every element are int64
    """
    reader = tf.TFRecordReader()

    # Read a single example
    _, image_file = reader.read(filename_queue)

    context_features = {
        "img": tf.FixedLenFeature([], tf.string)
    }
    sequence_features = {}
    for index in range(5):
        sequence_features["caption{}".format(index)] = tf.VarLenFeature(dtype=tf.float32)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=image_file,
        context_features=context_features,
        sequence_features=sequence_features
    )

    image = tf.decode_raw(context_parsed["img"], tf.uint8)
    image = tf.reshape(image, (64, 64, -1))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.cond(pred=tf.equal(tf.shape(image)[2], 3), fn2=lambda: tf.image.grayscale_to_rgb(image),
                    fn1=lambda: image)
    if flip:
        image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)

    if contrast:
        image = tf.image.random_contrast(image,
                                         lower=0.8, upper=1.2)
    # image = tf.image.random_brightness(image, max_delta=100)
    image.set_shape((64, 64, 3))
    inside_image = tf.image.central_crop(image, 0.50)
    inside_image.set_shape((32, 32, 3))

    # Compute mean color for each channel
    mean1 = tf.reduce_mean(image[:, :, 0])
    mean2 = tf.reduce_mean(image[:, :, 1])
    mean3 = tf.reduce_mean(image[:, :, 2])

    channel1 = tf.expand_dims(tf.constant(value=1.0, shape=(32, 32)) * mean1, dim=2)
    channel2 = tf.expand_dims(tf.constant(value=1.0, shape=(32, 32)) * mean2, dim=2)
    channel3 = tf.expand_dims(tf.constant(value=1.0, shape=(32, 32)) * mean3, dim=2)
    #
    mean_color = tf.stack([channel1, channel2, channel3], axis=2)
    mean_color = tf.squeeze(mean_color)

    # 1 * 1 in hole zone
    cropped_image = tf.ones(shape=tf.shape(inside_image))

    # Pad with surrouding zeros
    cropped_image = tf.pad(cropped_image, [[16, 16], [16, 16], [0, 0]])

    # Zero in the hole
    cropped_image = tf.subtract(1.0, cropped_image)

    # Fill with zeros value image
    cropped_image = image * cropped_image

    mean_color = tf.pad(mean_color, [[16, 16], [16, 16], [0, 0]])
    cropped_image += mean_color

    min_queue_examples = 256  # Shuffle elements

    caption0 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption0"]), (4800, 1))
    caption1 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption1"]), (4800, 1))
    caption2 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption2"]), (4800, 1))
    caption3 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption3"]), (4800, 1))
    caption4 = tf.reshape(tf.sparse_tensor_to_dense(sequence_parsed["caption4"]), (4800, 1))

    inputs = [image, cropped_image, inside_image,
              caption0,
              caption1,
              caption2,
              caption3,
              caption4]
    images = tf.train.shuffle_batch(
        inputs,
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    return images


def create_queue(filename, batch_size):
    """
    Create the queue, and get a holder on the queue input
    :return: list of SparseTensorValue and Tensor
        The output of a queue
    """
    path = os.path.join(os.path.dirname(os.path.basename(__file__)), "examples", "{}.tfrecords".format(filename))
    filename_queue = tf.train.string_input_producer(
        [path])
    return read_and_decode(filename_queue, batch_size)




def sample(mean, log_sigma):
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


if __name__ == '__main__':
    import numpy as np
    from tqdm import trange

    # Number of different captions to plot
    number_of_examples = 2000
    # Batch size (make sure gcd(number of examples, 5 *  batch_size) != 1
    batch_size = 20

    # If to saved them all temporary files are saved, else everything is collect and delete
    to_saved = False

    # Iterate over all the filename
    writer_filename = [os.path.join("examples", "train{}.tfrecords".format(i)) for i in range(3, 5)]
    filename_queue = tf.train.string_input_producer(
        writer_filename)
    images = read_and_decode(filename_queue, batch_size)

    # Dictionnary containing all pair of embedding-image
    dict = {}
    with tf.Session() as sess:
        group = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(group)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess)
        # Number of batches = number of different caption divided by 5 (because every image has 5 captions) and the batch size
        n_train_batches = number_of_examples // (5 * batch_size)

        # Iterate over all batches
        for i in trange(n_train_batches, leave=False):
            if coord.should_stop():
                break
            obj = sess.run([images])

        coord.join()
        coord.request_stop()
        coord.wait_for_stop()
