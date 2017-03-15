from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import tensorflow as tf
import os

__C = edict()
cfg = __C

# Object options
__C.obj = edict()
__C.obj.dic = {}

# Training options
__C.train = edict()
__C.train.batch_size = 16
__C.train.nb_epochs = 1500
__C.train.optimizer = tf.train.AdamOptimizer()

# Embedding values
__C.emb = edict()
__C.emb.nb_lang = 100
__C.emb.emb_dim = 200

# Generator values
__C.gen = edict()
__C.gen.z_dim = 100

# Queues options
__C.queue = edict()
__C.queue.filename = os.path.join(os.path.dirname(os.path.basename(__file__)), "examples", "train0.tfrecords")
__C.queue.is_val_set = False
__C.queue.nb_examples_per_file = 1000

# GAN options
__C.gan = edict()
__C.gan.train_adversarial = False
__C.gan.n_train_generator = 30

__C.gan.n_train_critic = 150
__C.gan.n_train_critic_intense = 300
__C.gan.intense_starting_period = 20
