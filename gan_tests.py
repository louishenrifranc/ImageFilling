import tensorflow as tf
from pprint import pprint

graph_saver = tf.train.import_meta_graph("model/model-80058.meta")
graph = tf.get_default_graph()

graph.get_operation_by_name("decoder/node5/")
sess = tf.Session(graph=graph)
print("Activation")
pprint([var.op.name for var in tf.get_collection(tf.GraphKeys.)])
print("Global variables")
pprint([var.op.name for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
print("Trainable variables")
pprint([var.op.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])


def _adversarial_loss(self, scope_name="discriminator"):
    real_logit = self._encoder(self.true_image, self.z, scope_name=scope_name)
    wrong_logit = self._encoder(self.wrong_image, self.z, scope_name=scope_name, reuse_variables=True)
    fake_logit = self._encoder(self.generated_image, self.z, scope_name=scope_name, reuse_variables=True)

    train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.gen_variables = [v for v in train_variables if not v.name.startswith("discriminator")]
    self.dis_variables = [v for v in train_variables if v.name.startswith("discriminator")]

    real_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_logit,
                                                                        tf.ones_like(real_logit)))
    wrong_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit,
                                                                         tf.zeros_like(wrong_logit)))

    fake_dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                                        tf.zeros_like(fake_logit)))
    self.dis_loss = real_dloss + (wrong_dloss + fake_dloss) / 2

    self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                                           tf.ones_like(fake_logit)))

    self.all_loss_G = self.gen_loss * 0.1 + (self.kl_loss + self._loss_recon_center
                                             + self._loss_recon_overlap) * 0.9
    self.all_loss_D = self.dis_loss

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
