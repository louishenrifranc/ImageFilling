import config
from model import Graph
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_first_file", type=int, default=3, help="Number of the first training file")
    parser.add_argument("--train_set_size", type=int, default=3000, help="Number of training examples")
    parser.add_argument("--val_set", type=bool, default=False, help="Whether to use a validation set")
    args = parser.parse_args()

    # Config default value
    cfg = config.cfg

    # Training files name
    cfg.queue.filename = ["train{}.tfrecords".format(index) for index in range(args.train_first_file,
                                                                               args.train_first_file +
                                                                               cfg.train_set_size //
                                                                               cfg.queue.nb_examples_per_file)]
    # Whether we create a validation set
    cfg.queue.is_val_set = args.val_set

    # Build model and train
    b = Graph(cfg)
    b.build()
    b.train()
