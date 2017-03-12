import tensorflow as tf
import numpy as np
import math


def rotate(image, angle):
    with tf.name_scope('rotate'):
        image = tf.cast(image, tf.float32)
        angle = angle / 180 * math.pi
        shape = image.get_shape().as_list()
        assert len(shape) == 3, "Input needs to be 3D."
        image_center = np.array([x / 2 for x in shape][:-1])

        coord1 = tf.cast(tf.range(shape[0]), tf.float32)
        coord2 = tf.cast(tf.range(shape[1]), tf.float32)

        # Create vectors of those coordinates in order to vectorize the image
        coord1_vec = tf.tile(coord1, [shape[1]])

        coord2_vec_unordered = tf.tile(coord2, [shape[0]])
        coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [shape[0], shape[1]])
        coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

        # center coordinates since rotation center is supposed to be in the image center
        coord1_vec_centered = coord1_vec - image_center[0]
        coord2_vec_centered = coord2_vec - image_center[1]

        coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

        # Perform backward transformation of the image coordinates
        rot_mat_inv = tf.expand_dims(tf.stack([tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), 0)
        rot_mat_inv = tf.cast(tf.reshape(rot_mat_inv, shape=[2, 2]), tf.float32)
        coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

        # Find neighbors in old image
        coord1_old_nn = coord_old_centered[0, :] + image_center[0]
        coord2_old_nn = coord_old_centered[1, :] + image_center[1]

        # Clip values to stay inside image coordinates
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, shape[0] - 1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, shape[1] - 1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

        # Coordinates of the new image
        coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

        coord1_old_nn0 = tf.floor(coord1_old_nn)
        coord2_old_nn0 = tf.floor(coord2_old_nn)
        sx = coord1_old_nn - coord1_old_nn0
        sy = coord2_old_nn - coord2_old_nn0
        coord1_old_nn0 = tf.cast(coord1_old_nn0, tf.int32)
        coord2_old_nn0 = tf.cast(coord2_old_nn0, tf.int32)
        coord1_old_nn0 = tf.boolean_mask(coord1_old_nn0, tf.logical_not(outside_ind))
        coord2_old_nn0 = tf.boolean_mask(coord2_old_nn0, tf.logical_not(outside_ind))
        coord1_old_nn1 = coord1_old_nn0 + 1
        coord2_old_nn1 = coord2_old_nn0 + 1
        interp_coords = [
            ((1. - sx) * (1. - sy), coord1_old_nn0, coord2_old_nn0),
            (sx * (1. - sy), coord1_old_nn1, coord2_old_nn0),
            ((1. - sx) * sy, coord1_old_nn0, coord2_old_nn1),
            (sx * sy, coord1_old_nn1, coord2_old_nn1)
        ]

        interp_old = []
        for intensity, coord1, coord2 in interp_coords:
            intensity = tf.transpose(tf.reshape(intensity, [shape[1], shape[0]]))
            coord_old_clipped = tf.transpose(tf.stack([coord1, coord2]), [1, 0])
            interp_old.append((intensity, coord_old_clipped))

        channels = tf.split(image, shape[2], 2)
        image_rotated_channel_list = list()
        for channel in channels:
            channel = tf.squeeze(channel)
            interp_intensities = []
            for intensity, coord_old_clipped in interp_old:
                image_chan_new_values = tf.gather_nd(channel, coord_old_clipped)

                channel_values = tf.sparse_to_dense(coord_new, [shape[0], shape[1]], image_chan_new_values,
                                                    0, validate_indices=False)

                interp_intensities.append(channel_values * intensity)
            image_rotated_channel_list.append(tf.add_n(interp_intensities))

        image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

        return image_rotated
