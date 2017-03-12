from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import trange
import tensorflow as tf
from PIL import Image
from utils import create_text_metadata
import numpy as np
import pickle
import os
import cv2


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

    image_unit32 = image
    # If number of channel is 1  -> modify to rgb scale
    image_unit32 = tf.cond(pred=tf.equal(tf.shape(image_unit32)[2], 3),
                           fn2=lambda: tf.image.grayscale_to_rgb(image_unit32),
                           fn1=lambda: image_unit32)

    # Need to define the true shape
    image_unit32.set_shape((64, 64, 3))

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

    # Number of different captions to plot
    number_of_examples = 6000

    # !!! IMPORTANT !!!
    # Not cross-OS function. File are not ordered in the same way between Linux and Windows
    # Make sure you assign the correct metadata to each file.
    # create_text_metadata(number_of_examples)

    # Batch size (make sure gcd(number of examples, 5 *  batch_size) != 1
    batch_size = 20

    # If to saved them all temporary files are saved, else everything is collect and delete
    to_saved = False

    # Iterate over all the filename
    writer_filename = [os.path.join("examples", "train{}.tfrecords".format(i)) for i in range(3, 10)]
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

            # Iterate over every batch element
            for b in range(batch_size):
                # Iterate over all captions of the image
                for k in range(1, 6):
                    # Check if the captions has not already been saved (yes it is terribly not optimal :) )
                    for v in dict.values():
                        if np.array_equal(v[0], obj[0][k][b]):
                            continue
                    # Save a pair of embedding - image
                    dict[len(dict)] = (obj[0][k][b].flatten(), obj[0][0][b])

        coord.join()
        coord.request_stop()
        coord.wait_for_stop()
    # Save embedding
    if to_saved:
        pickle.dump(dict, open("embedding.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    thumbnail_width = 28  # width of a small thumbnail
    thumbnail_height = thumbnail_width  # height

    # size of the embeddings
    embeddings_length = 4800

    # 1. Make the big spirit picture
    filename_spirit_picture = "master.jpg"
    filename_temporary_embedding = "features.p"
    dictionary = dict

    if not os.path.isfile(filename_spirit_picture) or not os.path.isfile(filename_temporary_embedding) or True:
        print("Creating spirit")
        Image.MAX_IMAGE_PIXELS = None
        images = []

        features = np.zeros((len(dictionary), embeddings_length))

        # Make a vector for all images and a list for their respective embedding (same index)
        for iteration, pair in dictionary.items():
            #
            array = cv2.resize(pair[1], (thumbnail_width, thumbnail_height))

            img = Image.fromarray(array)
            # Append the image to the list of images
            images.append(img)
            # Get the embedding for that picture
            features[iteration] = pair[0]

        # Build the spirit image
        print('Number of images %d' % len(images))
        image_width, image_height = images[0].size
        master_width = (image_width * (int)(np.sqrt(len(images))))
        master_height = master_width
        print('Length (in pixel) of the square image %d' % master_width)
        master = Image.new(
            mode='RGBA',
            size=(master_width, master_height),
            color=(0, 0, 0, 0))

        for count, image in enumerate(images):
            locationX = (image_width * count) % master_width
            locationY = image_height * (image_width * count // master_width)
            master.paste(image, (locationX, locationY))
        master.save(filename_spirit_picture, transparency=0)
        pickle.dump(features, open(filename_temporary_embedding, 'wb'))
    else:
        print('Spirit already created')
        features = pickle.load(open(filename_temporary_embedding, 'r'))

    print('Starting session')
    sess = tf.InteractiveSession()
    log_dir = 'logs'

    # Create a variable containing all features
    embeddings = tf.Variable(features, name='embeddings')

    # Initialize variables
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.save(sess, save_path=os.path.join(log_dir, 'model.ckpt'), global_step=None)

    # add metadata
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    metadata_path = os.path.join(log_dir, "metadata.tsv")
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.metadata_path = metadata_path

    print('Add metadata')
    embedding.tensor_name = embeddings.name

    # add image metadata
    embedding.sprite.image_path = filename_spirit_picture
    embedding.sprite.single_image_dim.extend([thumbnail_width, thumbnail_height])
    projector.visualize_embeddings(summary_writer, config)

    print('Finish now clean repo')
    # Clean actual repo
    if not to_saved:
        os.remove(filename_spirit_picture)
        os.remove(filename_temporary_embedding)
