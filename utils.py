import numpy as np
import pickle
import os


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
