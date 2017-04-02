import os
import tensorflow as tf
from tqdm import trange


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

    image = 2 * tf.image.convert_image_dtype(image, dtype=tf.float32) - 1

    image = tf.reshape(image, (64, 64, -1))
    image = tf.cond(pred=tf.equal(tf.shape(image)[2], 3), fn2=lambda: tf.image.grayscale_to_rgb(image),
                    fn1=lambda: image)
    if flip:
        image = tf.image.random_flip_left_right(image)

    image.set_shape((64, 64, 3))

    # image = tf.image.rgb_to_hsv(image)

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

    images = tf.train.batch(
        inputs,
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size)
    # min_after_dequeue=min_queue_examples)
    return images


def create_queue(filename, batch_size):
    """
    Create the queue, and get a holder on the queue input
    :return: list of SparseTensorValue and Tensor
        The output of a queue
    """
    filename_queue = tf.train.string_input_producer(
        filename)
    return read_and_decode(filename_queue, batch_size)


def sample(mean, log_sigma):
    epsilon = tf.truncated_normal(tf.shape(mean))
    # emb_dim (100)
    return epsilon * tf.exp(log_sigma) + mean


def get_mask_recon(hiding_size=32, overlap_size=7):
    mask_recon = tf.pad(tf.ones([hiding_size - 2 * overlap_size, hiding_size - 2 * overlap_size]),
                        [[overlap_size, overlap_size], [overlap_size, overlap_size]])
    mask_recon = tf.reshape(mask_recon, [hiding_size, hiding_size, 1])
    mask_recon = tf.concat([mask_recon] * 3, 2)
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
    pad_size = (image_size - hiding_size) // 2
    mask = tf.pad(tf.ones([hiding_size, hiding_size]), [[pad_size, pad_size], [pad_size, pad_size]])
    mask = tf.reshape(mask, [image_size, image_size, 1])
    mask = tf.concat([mask] * 3, 2)
    """
    ---------------
    |             |
    |     111     |
    |     111     |
    |             |
    ---------------
    """
    return mask


def reconstructed_image(reconstructed_hole, true_image):
    padded = tf.pad(tensor=reconstructed_hole, paddings=[[0, 0], [16, 16], [16, 16], [0, 0]])
    return padded + tf.stack([1 - get_mask_hiding()] * true_image.get_shape().as_list()[0],
                             axis=0) * true_image


def restore(model, save_name="model/", logs_folder="logs/"):
    """
    Retrieve last model saved if possible
    Create a main Saver object
    Create a SummaryWriter object
    Init variables
    :param save_name: string (default : model)
        Name of the model
    :return:
    """
    saver = tf.train.Saver(max_to_keep=1)
    # Try to restore an old model
    last_saved_model = tf.train.latest_checkpoint(save_name)

    group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    model.sess.run(group_init_ops)
    summary_writer = tf.summary.FileWriter(logs_folder,
                                           graph=model.sess.graph)
    if last_saved_model is not None:
        saver.restore(model.sess, last_saved_model)
        print("[*] Restoring model  {}".format(last_saved_model))
    else:
        tf.train.global_step(model.sess, model.global_step)
        print("[*] New model created")
    return saver, summary_writer


def train_epoch(model, saving_each_iter=10):
    nb_train_iter = (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size
    for i in trange(nb_train_iter, leave=False, desc="Training iteration"):
        op = [model.train_fn]
        if i % saving_each_iter == 0:
            op.append(model.merged_summary_op)
        out = model.sess.run(op, feed_dict={model.is_training: True})

        if i % saving_each_iter == 0:
            current_iter = model.sess.run(model.global_step)
            model.summary_writer.add_summary(out[1], global_step=current_iter)

    if not os.path.exists("model"):
        os.makedirs("model")
    current_iter = model.sess.run(model.global_step)
    model.saver.save(model.sess, "model/model", global_step=current_iter)


def train_adversarial_epoch(model, saving_each_iter=100):
    nb_train_iter = (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size
    print(nb_train_iter)
    for i in trange(nb_train_iter, leave=False, desc="Training iteration"):
        # op = model.train_gen
        # model.sess.run(op, feed_dict={model.is_training: True})

        op = [model.train_dis]
        if i % saving_each_iter == 0:
            op.append(model.merged_summary_op)
        out = model.sess.run(op, feed_dict={model.is_training: True})

        if i % saving_each_iter == 0:
            current_iter = model.sess.run(model.global_step)
            model.summary_writer.add_summary(out[1], global_step=current_iter)

    if not os.path.exists("model"):
        os.makedirs("model")
    current_iter = model.sess.run(model.global_step)
    model.saver.save(model.sess, "model/model", global_step=current_iter)


def compute_restart_epoch(model):
    current_step = model.global_step.eval(model.sess)

    if not model.adv_training:
        return current_step // (
            (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size)
    else:
        res = current_step // (
            (len(model.cfg.queue.filename) * model.cfg.queue.nb_examples_per_file) // model.batch_size)
        return res / 2
        # n_iter_starting = model.cfg.gan.n_train_critic_intense * model.cfg.gan.intense_starting_period \

#                   + model.cfg.gan.n_train_generator
# if current_step - n_iter_starting >= 0:
#     current_step -= n_iter_starting
#     n_iter = model.cfg.gan.n_train_critic + model.cfg.gan.n_train_generator
#     return current_step // n_iter
# else:
#     return current_step // n_iter_starting
