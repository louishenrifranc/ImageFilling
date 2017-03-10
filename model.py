import tensorflow as tf
import tensorflow.contrib.layers as ly
import helper
import tf_utils
from tqdm import trange


class Graph:
    def __init__(self, args):
        self.batch_size = args.train.batch_size
        self.nb_epochs = args.train.nb_epochs

        self.optimizer = args.train.optimizer
        # Placeholder
        self.is_training = tf.placeholder(tf.bool)

        # Global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Input filename
        self.cfg = args
        self.sess = tf.Session()

    def _inputs(self):
        input_to_graph = helper.create_queue(self.cfg.queue.filename, self.batch_size)
        # test_queue = helper.create_queue("val", self.batch_size)
        # input_graph = tf.cond(self.is_training, lambda: train_queue, lambda: test_queue)

        self.true_image = input_to_graph[0]
        self.cropped_image = input_to_graph[1]
        self.true_hole = input_to_graph[2]
        self.mean_caption = None
        for i in range(3, 8):
            input_to_graph[i] = tf.transpose(input_to_graph[i], [0, 2, 1])
            self.mean_caption = input_to_graph[i] if self.mean_caption is None else \
                tf.concat([self.mean_caption, input_to_graph[i]], axis=1)

        self.mean_caption = tf.reduce_mean(self.mean_caption, axis=1)

    def build(self):
        self._inputs()

        self._mean, self._log_sigma = self._generate_condition(self.mean_caption)

        # Sample conditioning from a Gaussian distribution parametrized by a Neural Network
        z = helper.sample(self._mean, self._log_sigma)

        # Encode the image
        z_vec = self._encoder(self.true_image, z)

        # Decode the image
        self.reconstructed_hole = self._decoder(z_vec)

        self._losses()
        self._optimize()
        self._summaries()

    def _generate_condition(self, sentence_embedding, scope_name="generate_condition", scope_reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if scope_reuse:
                scope.reuse_variables()
            out = ly.fully_connected(sentence_embedding,
                                     self.cfg.emb.emb_dim * 2,
                                     activation_fn=tf_utils.leaky_rectify)
            mean = out[:, :self.cfg.emb.emb_dim]
            log_sigma = out[:, self.cfg.emb.emb_dim:]
            # emb_dim
            return mean, log_sigma

    def _encoder(self, images, embedding, scope_name="encoder"):
        with tf.variable_scope(scope_name) as _:
            # Encode image
            # 32 * 32 * 64
            node1 = tf_utils.cust_conv2d(images, 64, h_f=4, w_f=4, batch_norm=False, scope_name="node1")
            # 16 * 16 * 128
            node1 = tf_utils.cust_conv2d(node1, 128, h_f=4, w_f=4, is_training=self.is_training, scope_name="node1_1")
            # 8 * 8 * 256
            node1 = tf_utils.cust_conv2d(node1, 128, h_f=4, w_f=4, is_training=self.is_training, scope_name="node1_2")
            # 4 * 4 * 256
            node1 = tf_utils.cust_conv2d(node1, 256, h_f=4, w_f=4, activation_fn=None, is_training=self.is_training,
                                         scope_name="node1_3")

            # 4 * 4 * 64
            node2 = tf_utils.cust_conv2d(node1, 64, h_f=1, w_f=1, h_s=1, w_s=1, is_training=self.is_training,
                                         scope_name="node2_1")
            # 4 * 4 * 128
            node2 = tf_utils.cust_conv2d(node2, 128, h_f=3, w_f=3, h_s=1, w_s=1, is_training=self.is_training,
                                         scope_name="node2_2")
            # 4 * 4 * 256
            node2 = tf_utils.cust_conv2d(node2, 256, h_f=3, w_f=3, h_s=1, w_s=1, activation_fn=None,
                                         is_training=self.is_training, scope_name="node2_3")

            # 4 * 4 * 256
            node = tf.add(node1, node2)
            node = tf_utils.leaky_rectify(node)

            # Encode embedding
            # 1 x 1 x nb_emb
            emb = tf.expand_dims(tf.expand_dims(embedding, 1), 1)
            # 4 x 4 x nb_emb
            emb = tf.tile(emb, [1, 4, 4, 1])

            # 4 x 4 x 356
            comb = tf.concat([node, emb], axis=3)

            # Compress embedding
            # 4 * 4 * 128
            result = tf_utils.cust_conv2d(comb, 128, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node3")
            return result

    def _decoder(self, input):
        # Node 0
        # 4 * 4 * 128
        node0_0 = tf_utils.cust_conv2d(input, 128, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node0")
        # 4 * 4 * 64
        node0_1 = tf_utils.cust_conv2d(node0_0, 64, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node0_1")
        # 4 * 4 * 64
        node0_1 = tf_utils.cust_conv2d(node0_1, 64, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node0_2")
        # 4 * 4 * 128
        node0_1 = tf_utils.cust_conv2d(node0_1, 128, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node0_3")

        # 4 * 4 * 128
        node1 = tf.add(node0_0, node0_1)

        # Node 1
        # 8 * 8 * 64
        node1_0 = tf_utils.cust_conv2d_transpose(node1, 64, w_s=2, h_s=2, is_training=self.is_training,
                                                 scope_name="node1_0")
        # 8 * 8 * 32
        node1_1 = tf_utils.cust_conv2d(node1_0, 32, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node1_1")
        # 8 * 8 * 32
        node1_1 = tf_utils.cust_conv2d(node1_1, 32, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node1_2")
        # 8 * 8 * 64
        node1_1 = tf_utils.cust_conv2d(node1_1, 64, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node1_3")

        # 8 * 8 * 64
        node2 = tf.add(node1_0, node1_1)

        # 8 * 8 * 32
        node2 = tf_utils.cust_conv2d(node2, 32, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                     scope_name="node2_0")

        # Node 2
        # 16 * 16 * 16
        node2_0 = tf_utils.cust_conv2d_transpose(node2, 16, h_s=2, w_s=2, is_training=self.is_training,
                                                 scope_name="node2_1")
        # 16 * 16 * 8
        node2_1 = tf_utils.cust_conv2d(node2_0, 8, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node2_2")
        # 16 * 16 * 8
        node2_1 = tf_utils.cust_conv2d(node2_1, 8, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node2_3")
        # 16 * 16 * 16
        node2_1 = tf_utils.cust_conv2d(node2_1, 16, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                       scope_name="node2_4")

        node3 = tf.add(node2_0, node2_1)

        # Node 3
        # 32 x 32 x 8
        node3_0 = tf_utils.cust_conv2d_transpose(node3, 8, h_s=2, w_s=2, is_training=self.is_training,
                                                 scope_name="node3")

        # 32 x 32 x 3
        out = tf_utils.cust_conv2d(node3_0, 3, h_f=1, w_f=1, w_s=1, h_s=1, activation_fn=tf.tanh,
                                   is_training=self.is_training, scope_name="node4")
        return out

    def _losses(self):
        # KL loss
        self.kl_loss = -self._log_sigma + 0.5 * (-1 + tf.exp(2 * self._log_sigma) + tf.square(self._mean))
        self.kl_loss = tf.reduce_mean(self.kl_loss)

        # Reconstruction error
        recon_mask = helper.get_mask_recon()
        # Loss for original image
        loss_recon_ori = tf.square(self.true_hole - self.reconstructed_hole)
        self._loss_recon_center = tf.reduce_mean(
            tf.sqrt(1e-5 + tf.reduce_sum(loss_recon_ori * (1 - recon_mask), [1, 2, 3])))
        self._loss_recon_overlap = tf.reduce_mean(
            tf.sqrt(1e-5 + tf.reduce_sum(loss_recon_ori * recon_mask, [1, 2, 3]))) * 10
        self.loss = self._loss_recon_center + self._loss_recon_overlap + self.kl_loss

    def _optimize(self):
        """
        Helper to create mechanism for computing the derivative wrt to the loss
        :return:
        """
        # Retrieve all trainable variables
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Compute the gradient (return a pair of variable and their respective gradient)
        grads = self.optimizer.compute_gradients(loss=self.loss, var_list=train_variables)
        self.train_fn = self.optimizer.apply_gradients(grads, global_step=self.global_step)

    def _summaries(self):
        """
        Helper to add summaries
        :return:
        """
        trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variable:
            tf.summary.histogram(var.op.name, var)

        # Add summaries for images
        tf.summary.image(name="true_image", tensor=self.true_image)
        tf.summary.image(name="crop_image", tensor=self.cropped_image)
        tf.summary.image(name="true_hole", tensor=self.true_hole)
        tf.summary.image(name="reconstructed_hole", tensor=self.reconstructed_hole)

        # Add summaries for loss functions
        tf.summary.scalar(name="loss_recon_center", tensor=self._loss_recon_center)
        tf.summary.scalar(name="loss_recon_overlap", tensor=self._loss_recon_overlap)
        tf.summary.scalar(name="kl_loss", tensor=self.kl_loss)
        tf.summary.scalar(name="loss", tensor=self.loss)

        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        self.saver, self.summary_writer = helper.restore(self.sess)

        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        tf.train.start_queue_runners(sess=self.sess)

        coord = tf.train.Coordinator()
        for _ in trange(self.nb_epochs, desc="Epoch"):
            if coord.should_stop():
                break

            helper.train_epoch(self)
            if self.cfg.queue.is_val_set:
                # TODO: waiting for validation embedding created
                pass

        coord.request_stop()
        coord.join()
