import pickle
import os


def create_text_metadata(nb_examples, starting_index=3000, nb_caption_per_picture=5):
    """
    Number of examples to plot
    :param nb_examples: Integer
    :return:
    """
    # Path where to save the metadata
    path_logs = "logs"
    # Path of the directory containing the images to plot
    path_images = "train2014"
    # Path containing the directory containing all the dict
    caption_file = pickle.load(open("dict.pkl", "rb"))

    metadata = os.path.join(path_logs, 'metadata.tsv')
    nb_examples /= nb_caption_per_picture
    # Iterate over all files
    with open(metadata, 'w') as metadata_file:
        metadata_file.write("Caption\tFilename\n")
        for index, filename in enumerate(os.listdir(path_images)):
            if index < starting_index:
                continue
            if index >= (starting_index + nb_examples):
                break
            captions = caption_file[filename.split(".")[0]]
            for i, caption in enumerate(captions):
                if i >= nb_caption_per_picture:
                    break
                metadata_file.write('%s\t%s\n' % ("_".join(caption.split(" ")), filename))


def create_dictionnary():
    """
    Create a dictionnary which map word from word to Glove embedding
    :return:
    """
    # Path of the glove embedding
    glove_path = os.path.join(os.path.dirname(os.path.basename(__file__)), "dictionnary", "glove.6B.300d.txt")

    # Vocab dic
    dic_path = os.path.join(os.path.dirname(os.path.basename(__file__)), "worddict.pkl")
    dic = pickle.load(open(dic_path, "rb"))

    # word 2 embedding dictionnary
    word2idx_path = os.path.join(os.path.dirname(os.path.basename(__file__)), "dictionnary", "word2idx.pkl")
    idx2emb_path = os.path.join(os.path.dirname(os.path.basename(__file__)), "dictionnary", "idx2emb.pkl")

    word2idx = {}
    idx2emb = {}
    with open(glove_path) as f:
        line = f.readline()
        while line:
            if line[0] == " ":
                pass
            else:
                l = line.split(" ")
                if l[0] in dic:
                    word2idx[l[0]] = len(idx2emb)
                    idx2emb[len(idx2emb)] = [float(n) for n in l[1:]]
            line = f.readline()
    print("Found embedding ratio: {}".format(len(word2idx) / len(dic)))
    pickle.dump(word2idx, open(word2idx_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(idx2emb, open(idx2emb_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


# create_dictionnary()
def back(filename):
    return os.path.join("..", filename)


def get_max_len_caption():
    import pickle
    caption_file = pickle.load(open(back("dict.pkl"), "rb"))
    max_l = 0
    for _, c in caption_file.items():
        for s in c:
            max_l = max(max_l, len(s.split()))
    return max_l
