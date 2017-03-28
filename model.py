import tensorflow as tf
import tensorflow.contrib.layers as ly
import helper
import tf_utils
from tqdm import trange


class Graph:
    def __init__(self, args):
        self.batch_size = args.train.batch_size
        self.nb_epochs = args.train.nb_epochs

        self.adv_training = args.gan.train_adversarial

        self.optimizer = args.train.optimizer
        # Placeholder
        self.is_training = tf.placeholder(tf.bool)

        # Global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.decay_l2_loss = tf.train.exponential_decay(1.0, self.global_step,
                                                        20000, 0.95, staircase=True)

        self.dropout_ratio = tf.train.exponential_decay(1.0, self.global_step,
                                                        20000, 0.95, staircase=True)

        # self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step,
        #                                                 20000, 0.98, staircase=True)
        # Input filename
        self.cfg = args
        self.sess = tf.Session()

    def _inputs(self):
        input_to_graph = helper.create_queue(self.cfg.queue.filename, self.batch_size)
        if self.cfg.queue.is_val_set:
            # TODO: not implemented + using tf.cond is bad
            test_queue = helper.create_queue("val", self.batch_size)
            input_to_graph = tf.cond(self.is_training, lambda: input_to_graph, lambda: test_queue)

        # True image
        self.true_image = input_to_graph[0]

        # True image filled with mean color (64 x 64 x 3)
        self.cropped_image = input_to_graph[1]

        # True hole (32 x 32 x 3)
        self.true_hole = input_to_graph[2]

        # Wrong image (for adversarial cost) (64 x 64 x 3)
        self.wrong_image = input_to_graph[8]

        # Mean of the caption
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
        self.z = helper.sample(self._mean, self._log_sigma)

        # Encode the image
        self.z_vec = self._encoder(self.true_image, self.z)

        # Decode the image
        self.reconstructed_hole = self._decoder(self.z_vec)
        self.generated_image = helper.reconstructed_image(self.reconstructed_hole, self.true_image)
        self._losses()
        self._adversarial_loss()
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

    def _encoder(self, images, embedding, scope_name="encoder", reuse_variables=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse_variables:
                scope.reuse_variables()

            # images = ly.dropout(images, keep_prob=self.dropout_ratio, is_training=self.is_training)
            # Encode image
            # 32 * 32 * 64
            node1 = tf_utils.cust_conv2d(images, 64, h_f=4, w_f=4, batch_norm=False, scope_name="node1")
            # 16 * 16 * 128
            node1 = tf_utils.cust_conv2d(node1, 128, h_f=4, w_f=4, is_training=self.is_training, scope_name="node1_1")
            # 8 * 8 * 256
            node1 = tf_utils.cust_conv2d(node1, 256, h_f=4, w_f=4, is_training=self.is_training, scope_name="node1_2")
            # 4 * 4 * 512
            node1 = tf_utils.cust_conv2d(node1, 512, h_f=4, w_f=4, activation_fn=None, is_training=self.is_training,
                                         scope_name="node1_3")

            # node1 = ly.dropout(node1, keep_prob=self.dropout_ratio, is_training=self.is_training)

            # 4 * 4 * 128
            node2 = tf_utils.cust_conv2d(node1, 256, h_f=1, w_f=1, h_s=1, w_s=1, is_training=self.is_training,
                                         scope_name="node2_1")
            # 4 * 4 * 128
            node2 = tf_utils.cust_conv2d(node2, 256, h_f=3, w_f=3, h_s=1, w_s=1, is_training=self.is_training,
                                         scope_name="node2_2")
            # 4 * 4 * 512
            node2 = tf_utils.cust_conv2d(node2, 512, h_f=3, w_f=3, h_s=1, w_s=1, activation_fn=None,
                                         is_training=self.is_training, scope_name="node2_3")

            # node2 = ly.dropout(node2, keep_prob=self.dropout_ratio, is_training=self.is_training)
            # 4 * 4 * 512
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
            # 4 * 4 * 256
            result = tf_utils.cust_conv2d(comb, 512, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node3")
            result = tf_utils.cust_conv2d(result, 256, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node4")
            if scope_name == "discriminator":
                result = tf_utils.cust_conv2d(result, 128, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node5")
                # result = tf_utils.cust_conv2d(result, 64, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node6")

                # result = tf_utils.cust_conv2d(result, 1, h_f=3, w_f=3, w_s=1, h_s=1, scope_name="node7")
            return result

    def _decoder(self, input, scope_name="decoder"):
        with tf.variable_scope(scope_name) as _:
            # Node 0
            # 4 * 4 * 256
            node0_0 = tf_utils.cust_conv2d(input, 256, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node0")
            # 4 * 4 * 64
            node0_1 = tf_utils.cust_conv2d(node0_0, 128, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node0_1")
            # 4 * 4 * 64
            node0_1 = tf_utils.cust_conv2d(node0_1, 128, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node0_2")
            # 4 * 4 * 128
            node0_1 = tf_utils.cust_conv2d(node0_1, 256, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node0_3")

            # 4 * 4 * 128
            node1 = tf.add(node0_0, node0_1)

            # Node 1
            # 8 * 8 * 64
            node1_0 = tf_utils.cust_conv2d_transpose(node1, 128, w_s=2, h_s=2, is_training=self.is_training,
                                                     scope_name="node1_0")
            # 8 * 8 * 32
            node1_1 = tf_utils.cust_conv2d(node1_0, 64, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node1_1")
            # 8 * 8 * 32
            node1_1 = tf_utils.cust_conv2d(node1_1, 64, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node1_2")
            # 8 * 8 * 64
            node1_1 = tf_utils.cust_conv2d(node1_1, 128, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node1_3")

            # 8 * 8 * 64
            node2 = tf.add(node1_0, node1_1)

            # Node 2
            # 16 * 16 * 16
            node2_0 = tf_utils.cust_conv2d_transpose(node2, 64, h_s=2, w_s=2, is_training=self.is_training,
                                                     scope_name="node2_1")
            # 16 * 16 * 8
            node2_1 = tf_utils.cust_conv2d(node2_0, 32, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node2_2")
            # 16 * 16 * 8
            node2_1 = tf_utils.cust_conv2d(node2_1, 32, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node2_3")
            # 16 * 16 * 16
            node2_1 = tf_utils.cust_conv2d(node2_1, 64, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node2_4")

            node3 = tf.add(node2_0, node2_1)

            # Node 3
            # 32 x 32 x 8
            node3_0 = tf_utils.cust_conv2d_transpose(node3, 32, h_s=2, w_s=2, is_training=self.is_training,
                                                     scope_name="node3")
            # 16 * 16 * 8
            node3_1 = tf_utils.cust_conv2d(node3_0, 16, h_f=1, w_f=1, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node3_1")
            # 16 * 16 * 8
            node3_1 = tf_utils.cust_conv2d(node3_1, 16, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node3_2")
            # 16 * 16 * 16
            node3_1 = tf_utils.cust_conv2d(node3_1, 32, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node3_3")

            node4 = tf.add(node3_0, node3_1)

            node4_1 = tf_utils.cust_conv2d(node4, 16, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node4_1")
            node4_1 = tf_utils.cust_conv2d(node4_1, 8, h_f=3, w_f=3, w_s=1, h_s=1, is_training=self.is_training,
                                           scope_name="node4_2")

            # 32 x 32 x 3
            out = tf_utils.cust_conv2d(node4_1, 3, h_f=1, w_f=1, w_s=1, h_s=1, activation_fn=tf.tanh,
                                       is_training=self.is_training, scope_name="node5")
            return out

    def _adversarial_loss(self, scope_name="discriminator"):
        real_logit = self._encoder(self.true_image, self.z, scope_name=scope_name)
        wrong_logit = self._encoder(self.wrong_image, self.z, scope_name=scope_name, reuse_variables=True)
        fake_logit = self._encoder(self.generated_image, self.z, scope_name=scope_name, reuse_variables=True)

        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gen_variables = [v for v in train_variables if not v.name.startswith("discriminator")]
        from pprint import pprint
        pprint([var.op.name for var in self.gen_variables])

        self.dis_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        pprint([var.op.name for var in self.dis_variables])

        real_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                                            labels=tf.ones_like(real_logit)))
        wrong_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                                             labels=tf.zeros_like(wrong_logit)))

        fake_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                                            labels=tf.zeros_like(fake_logit)))
        self.dis_loss = real_dloss + (wrong_dloss + fake_dloss) / 2

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                                               labels=tf.ones_like(fake_logit)))

        self.all_loss_G = 10 * self.gen_loss * (1 - self.decay_l2_loss) + (self.kl_loss + self._loss_recon_center
                                                                           + self._loss_recon_overlap) * self.decay_l2_loss
        self.all_loss_D = 10 * self.dis_loss

        W_G = filter(lambda x: x.name.endswith('weights:0'), self.gen_variables)
        W_D = filter(lambda x: x.name.endswith('weights:0'), self.dis_variables)

        self.all_loss_G += 0.00001 * tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in W_G]))
        self.all_loss_D += 0.00001 * tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in W_D]))

        grads_var_dis = self.optimizer.compute_gradients(loss=self.dis_loss, var_list=self.dis_variables)
        grads_var_dis = map(lambda gv: [tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]], grads_var_dis)
        self.train_dis = self.optimizer.apply_gradients(grads_var_dis, global_step=self.global_step)

        grads_var_gen = self.optimizer.compute_gradients(loss=self.gen_loss, var_list=self.gen_variables)
        grads_var_gen = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_var_gen)
        self.train_gen = self.optimizer.apply_gradients(grads_var_gen, global_step=self.global_step)

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

        # Add summaries for images
        num_images = self.batch_size
        tf.summary.image(name="crop_image", tensor=self.cropped_image, max_outputs=num_images)
        tf.summary.image(name="true_hole", tensor=self.true_hole, max_outputs=num_images)
        tf.summary.image(name="reconstructed_hole", tensor=self.reconstructed_hole, max_outputs=num_images)
        tf.summary.image(name="true_image", tensor=self.true_image, max_outputs=num_images)
        tf.summary.image(name="reconstructed_image", tensor=self.generated_image, max_outputs=num_images)

        # Add summaries for loss functions
        tf.summary.scalar(name="loss_recon_center", tensor=self._loss_recon_center)
        tf.summary.scalar(name="loss_recon_overlap", tensor=self._loss_recon_overlap)
        tf.summary.scalar(name="kl_loss", tensor=self.kl_loss)
        tf.summary.scalar(name="loss", tensor=self.loss)
        tf.summary.scalar(name="generator_loss", tensor=self.gen_loss)
        tf.summary.scalar(name="discriminator_loss", tensor=self.dis_loss)
        tf.summary.scalar(name="full_discriminator_loss", tensor=self.all_loss_D)
        tf.summary.scalar(name="full_generator_loss", tensor=self.all_loss_G)

        self.merged_summary_op = tf.summary.merge_all()

    def train(self):

        self.saver, self.summary_writer = helper.restore(self)

        tf.train.start_queue_runners(sess=self.sess)

        coord = tf.train.Coordinator()

        train_fn = helper.train_adversarial_epoch if self.adv_training else helper.train_epoch

        epoch_restart = helper.compute_restart_epoch(self)
        print(epoch_restart)
        for self.epoch in trange(self.nb_epochs, desc="Epoch"):
            if coord.should_stop():
                break
            if self.epoch < epoch_restart:
                continue
            train_fn(self)
            if self.cfg.queue.is_val_set:
                # TODO: waiting for validation embedding created
                pass

        coord.request_stop()
        coord.join()

    def fill_image(self, num_images):
        _, self.summary_writer = helper.restore(self, logs_folder="prediction/")

        tf.train.start_queue_runners(sess=self.sess)

        self.batch_size = num_images
        _, summary_str = self.sess.run([self.generated_image, self.merged_summary_op],
                                       feed_dict={self.is_training: False})
        
        self.summary_writer.add_summary(summary_str)
        self.summary_writer.flush()
