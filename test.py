import os
import pickle


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


create_text_metadata(6000)
