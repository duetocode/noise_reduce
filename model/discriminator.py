import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, flatten, batch_norm
from tensorflow.nn import leaky_relu
from .norms import pixel_norm

def build_discriminator(images, layer_num=6):
    with tf.variable_scope('discriminator'):
        net = tf.identity(images, name='input')

        filter_base = 16
        for i in range(layer_num):
            residual = net
            net = conv2d(net, filter_base * 2**i, kernel_size=1, activation_fn=leaky_relu, normalizer_fn=batch_norm)
            net = conv2d(net, filter_base * 2**(i+1), kernel_size=3, stride=2, activation_fn=leaky_relu, normalizer_fn=batch_norm)
            print(f'{i+1}: {residual.shape.as_list()[1:]} -> {net.shape.as_list()[1:]}')
        
        net = flatten(net)
        net = fully_connected(net, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

        return tf.identity(net, name='output')