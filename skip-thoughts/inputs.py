import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange, tqdm
from PIL import Image
import numpy as np
import utils
import pickle
import os

file_path = os.path.dirname(os.path.abspath(__file__))


def make_example(img, captions):
    ex = tf.train.SequenceExample()
    ex.context.feature["img"].bytes_list.value.append(img)
    for index, caption in enumerate(captions):
        fl_list = ex.feature_lists.feature_list["caption{}".format(index)]
        for w in caption:
            fl_list.feature.add().float_list.value.append(w)
    return ex


def build_examples():
    import skipthoughts
    model = skipthoughts.load_model()

    caption_file = pickle.load(open(os.path.join(file_path, utils.back("dict.pkl")), "rb"))

    path_to_save_examples = os.path.join(file_path, utils.back("examples"))
    if not os.path.exists(path_to_save_examples):
        os.makedirs(path_to_save_examples)

    nb_file_per_tfrecords = 1000
    for name in ["train", "val"]:
        # Path where images are
        path = os.path.join(file_path, utils.back("{}2014".format(name)))
        # Number of images in the folder
        nb_file = len([n for n in os.listdir(path)])

        # Number of tfRecords already created
        nb_records = len([n for n in os.listdir(path_to_save_examples) if n.startswith(name)])
        # The file number to restart from
        iter_to_restart = (nb_records - 1) * nb_file_per_tfrecords
        iter_to_restart = 79000
        for iter in trange(iter_to_restart, nb_file, nb_file_per_tfrecords):
            writer = tf.python_io.TFRecordWriter(
                os.path.join(path_to_save_examples, "{}{}".format(name, iter // nb_file_per_tfrecords) + ".tfrecords"))
            for index, filename in tqdm(enumerate(os.listdir(path)), leave=False,
                                        desc="TfRecord{}".format(iter // nb_file_per_tfrecords)):
                if index < iter:
                    continue
                if index >= iter + nb_file_per_tfrecords:
                    break

                img = np.array(Image.open(os.path.join(path, filename)))
                caption = caption_file[filename.split(".")[0]]
                emb = skipthoughts.encode(model, caption, verbose=False)
                img = img.tostring()
                ex = make_example(img, emb)
                writer.write(ex.SerializeToString())

            writer.close()
            break




if __name__ == '__main__':
    build_examples()
