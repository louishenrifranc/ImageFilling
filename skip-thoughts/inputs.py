import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange, tqdm
from PIL import Image
import numpy as np
import utils
import pickle
import os

file_path = os.path.dirname(os.path.abspath(__file__))


def make_example(img, captions):
    ex = tf.train.SequenceExample()
    ex.context.feature["img"].bytes_list.value.append(img)
    for index, caption in enumerate(captions):
        fl_list = ex.feature_lists.feature_list["caption{}".format(index)]
        for w in caption:
            fl_list.feature.add().float_list.value.append(w)
    return ex


def build_examples():
    import skipthoughts
    model = skipthoughts.load_model()

    caption_file = pickle.load(open(os.path.join(file_path, utils.back("dict.pkl")), "rb"))

    path_to_save_examples = os.path.join(file_path, utils.back("examples"))
    if not os.path.exists(path_to_save_examples):
        os.makedirs(path_to_save_examples)

    nb_file_per_tfrecords = 1000
    for name in ["train", "val"]:
        # Path where images are
        path = os.path.join(file_path, utils.back("{}2014".format(name)))
        # Number of images in the folder
        nb_file = len([n for n in os.listdir(path)])

        # Number of tfRecords already created
        nb_records = len([n for n in os.listdir(path_to_save_examples) if n.startswith(name)])
        # The file number to restart from
        iter_to_restart = (nb_records - 1) * nb_file_per_tfrecords

        for iter in trange(iter_to_restart, nb_file, nb_file_per_tfrecords):
            writer = tf.python_io.TFRecordWriter(
                os.path.join(path_to_save_examples, "{}{}".format(name, iter // nb_file_per_tfrecords) + ".tfrecords"))
            for index, filename in tqdm(enumerate(os.listdir(path)), leave=False,
                                        desc="TfRecord{}".format(iter // nb_file_per_tfrecords)):
                if index < iter:
                    continue
                if index >= iter + nb_file_per_tfrecords:
                    break

                img = np.array(Image.open(os.path.join(path, filename)))
                caption = caption_file[filename.split(".")[0]]
                emb = skipthoughts.encode(model, caption, verbose=False)
                img = img.tostring()
                ex = make_example(img, emb)
                writer.write(ex.SerializeToString())

            writer.close()


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

    caption0 = sequence_parsed["caption0"]
    caption1 = sequence_parsed["caption1"]
    caption2 = sequence_parsed["caption2"]
    caption3 = sequence_parsed["caption3"]
    caption4 = sequence_parsed["caption4"]

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


if __name__ == '__main__':
    build_examples()
