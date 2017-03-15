import config
from model import Graph
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_first_file", type=int, default=3, help="Number of the first training file")
    parser.add_argument("--train_set_size", type=int, default=1000, help="Number of training examples")
    parser.add_argument("--val_set", type=bool, default=False, help="Whether to use a validation set")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of a batch")
    parser.add_argument("--train_adversarial", type=bool, default=False,
                        help="Whether to train the model adding an adversarial cost")
    parser.add_argument("--train", action="store_true", default=True,
                        help="If the model should be trained")

    args = parser.parse_args()
    print(args)
    # Config default value
    cfg = config.cfg

    # Training files name
    cfg.queue.filename = [
        os.path.join(os.path.dirname(os.path.basename(__file__)), "examples", "train{}.tfrecords").format(index)
        for index in range(args.train_first_file,
                           args.train_first_file +
                           args.train_set_size //
                           cfg.queue.nb_examples_per_file)]

    print(cfg.queue.filename)

    # Whether we create a validation set
    cfg.queue.is_val_set = args.val_set

    # Whether to train with adversarial cost
    cfg.gan.train_adversarial = args.train_adversarial

    # Size of a batch
    cfg.train.batch_size = args.batch_size

    # Build model and train or fill images
    b = Graph(cfg)
    b.build()
    if args.train:
        b.train()
    else:
        # TODO: add a queue for validation set (change args parameter in consequences)
        b.fill_image(20)
