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


def cust_conv2d(input_layer, out_dim, in_dim=None, h_f=3, w_f=3, h_s=2, w_s=2, padding="SAME", scope_name="conv2D",
                batch_norm=True, activation_fn=tf_utils.leaky_rectify):
    with tf.variable_scope(scope_name) as _:
        w = tf.Variable(name='W', initial_value=tf.truncated_normal_initializer(
            [h_f, w_f, in_dim or input_layer.shape[-1], out_dim], stddev=0.02))
        out = tf.nn.conv2d(input_layer.tensor, w, strides=[1, h_s, w_s, 1], padding=padding)
        if batch_norm:
            out = ly.batch_norm(out)
        if activation_fn is not None:
            out = activation_fn(out)
        return out


def cust_conv2d_transpose(input_layer, out_dim, h_f=3, w_f=3, h_s=2, w_s=2, padding="SAME", scope_name="conv2d_trans",
                          batch_norm=True, activation_fn=tf_utils.leaky_rectify)
    with tf.variable_scope(scope_name) as _:
        out = ly.conv2d_transpose(input_layer, out_dim, [h_f, w_f], stride=[h_s, w_s], padding=padding)
        if batch_norm:
            out = ly.batch_norm(out)
        if activation_fn is not None:
            out = activation_fn(out)
        return out


def channel_wise_fc(input_layer):
    return cust_conv2d(input_layer, input_layer.shape[-1], h_f=1, w_f=1, h_s=1, w_s=1, batch_norm=False,
                       activation_fn=None)
