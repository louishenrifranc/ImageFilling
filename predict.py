import tensorflow as tf

def read_and_decode(filename_queue,
                    batch_size,
                    flip=True,
                    wrong_image=True):
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

    if wrong_image:
        wrong_images = tf.train.shuffle_batch(
            [image],
            batch_size=1,
            capacity=min_queue_examples + 3,
            min_after_dequeue=min_queue_examples)
        inputs.append(wrong_images[0])

    images = tf.train.shuffle_batch(
        inputs,
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    return images
