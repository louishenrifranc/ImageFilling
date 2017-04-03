import tensorflow as tf
from pprint import pprint
from tkinter.filedialog import askopenfilename
import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.transform import resize


def retrieve_image(filename="images/batch_norm.jpg"):
    img = plt.imread(filename)
    # Between 0 and 1
    img = 2 * skimage.img_as_float(img) - 1
    new_img = resize(img, (64, 64), preserve_range=True)
    mean_color = np.mean(np.reshape(new_img, (-1, 3)), axis=0)
    new_img[16:48, 16:48, :] = mean_color
    input_img = np.repeat(np.expand_dims(new_img, 0), 16, 0)

    emb_input = np.random.normal(1e-5, 0.01, (16, 4800))
    return input_img, emb_input


filename = askopenfilename()

img, emb = retrieve_image(filename)
saver = tf.train.import_meta_graph('model/model-234745.meta')
graph = tf.get_default_graph()

is_training = graph.get_tensor_by_name('Placeholder:0')
cropped_img = graph.get_tensor_by_name('batch:0')
mean_caption = graph.get_tensor_by_name('Mean_3:0')
reconstructed_img = graph.get_tensor_by_name("add_2:0")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

out = reconstructed_img.eval(session=sess, feed_dict={is_training: False, cropped_img: img, mean_caption: emb})
plt.imshow(out[0])
plt.show()
