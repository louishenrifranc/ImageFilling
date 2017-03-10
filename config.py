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
__C.train.batch_size = 32
__C.train.nb_epochs = 600
__C.train.optimizer = tf.train.AdamOptimizer()

# Embedding values
__C.emb = edict()
__C.emb.nb_lang = 100
__C.emb.emb_dim = 100

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
