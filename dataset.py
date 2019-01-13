import tensorflow as tf
import numpy as np
from tensorflow.data import TFRecordDataset

height = 218
width = 178

def decode_image(example):
    features = tf.parse_single_example(example, {
        "img": tf.FixedLenFeature((), tf.string, default_value="")})
    image = tf.image.decode_jpeg(features['img'])
    image = tf.cast(image, tf.float32) / 255.0 * 2.0 - 1.0
    image.set_shape((height, width, 3))
    return image

def open_new_dataset(file='dataset/dataset.tr', shuffle_buffer=512, batch_size=2):
    '''Open a new dataset'''
    result = TFRecordDataset(file) \
        .shuffle(shuffle_buffer) \
        .map(decode_image) \
        .batch(batch_size)
        
    return result

def wreck(images):
    return images * np.random.rand(*images.shape)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    tf.reset_default_graph()
    dataset = open_new_dataset()
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()

    with tf.Session() as sess:
        original = sess.run(images)
        print(original.shape)
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(original[0])
        fig.add_subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(wreck(original[0]))
        plt.show()