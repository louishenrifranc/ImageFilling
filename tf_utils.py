import tensorflow as tf
import tensorflow.contrib.layers as ly
import tf_utils


def length(sequence):
    """
    @sequence: 3D tensor of shape (batch_size, sequence_length, embedding_size)
    """
    used = tf.sign(tf.reduce_sum(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length  # vector of size (batch_size) containing sentence lengths


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def leaky_rectify(x, leakiness=0.2):
    return tf.maximum(x, leakiness * x)


def cust_conv2d(input_layer, out_dim, h_f=3, w_f=3, h_s=2, w_s=2, padding="SAME", scope_name=None,
                batch_norm=True, activation_fn=tf_utils.leaky_rectify, is_training=True):
    with tf.variable_scope(scope_name) as _:
        out = ly.conv2d(input_layer,
                        out_dim,
                        [w_f, h_f],
                        [h_s, w_s],
                        padding,
                        activation_fn=None)
        if batch_norm:
            out = ly.batch_norm(out, is_training=is_training)
        if activation_fn:
            out = activation_fn(out)
        return out


def cust_conv2d_transpose(input_layer, out_dim, h_f=3, w_f=3, h_s=2, w_s=2, padding="SAME",
                          scope_name="transpose_conv_2D",
                          batch_norm=True, activation_fn=tf_utils.leaky_rectify, is_training=True):
    with tf.variable_scope(scope_name) as _:
        out = ly.conv2d_transpose(input_layer,
                                  out_dim,
                                  [w_f, h_f],
                                  [w_s, h_s],
                                  padding,
                                  activation_fn=None)
        if batch_norm:
            out = ly.batch_norm(out, is_training=is_training)
        if activation_fn:
            out = activation_fn(out)
        return out


def channel_wise_fc(input_layer):
    return cust_conv2d(input_layer, input_layer.shape[-1], h_f=1, w_f=1, h_s=1, w_s=1, batch_norm=False,
                       activation_fn=None)


def add_summary_python_scalar(name, scalar):
    return tf.Summary(value=[
        tf.Summary.Value(tag=name, simple_value=scalar),
    ])
