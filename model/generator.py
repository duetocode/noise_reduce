import tensorflow as tf
from tensorflow.contrib.layers import conv2d, conv2d_transpose, batch_norm
from .norms import pixel_norm

def build_generator(input_tensor, layer_num=6):
    with tf.variable_scope('generator'):
        net = tf.identity(input_tensor, name='input')
        # Conv Part
        net, residuals = build_conv2d_part(input_tensor, layer_num)
        # Deconv Part
        net = build_deconv2d_part(net, residuals)
        return tf.identity(net, name='output')

def build_conv2d_part(input_tensor, layer_num=5, filters_base = 16):
    residuals = []
    
    net = conv2d(input_tensor, filters_base, kernel_size=1, normalizer_fn=batch_norm)
    
    for i in range(1, layer_num+1):
        # every layer will down scale by 2
        with tf.variable_scope(f'block-down-{i}'):
            residual = net
            residuals.append(residual)
            net = conv2d(net, filters_base * 2**i, kernel_size=1, padding='SAME', normalizer_fn=batch_norm)
            net = conv2d(net, filters_base * 2**i, kernel_size=3, stride=2, padding='SAME', normalizer_fn=batch_norm)
            print(f'{i}: {residual.shape[1:]} -> {net.shape[1:]}')
    return net, residuals

def build_deconv2d_part(input_tensor, residuals):
    net = input_tensor
    for i, residual in enumerate(reversed(residuals)):
        with tf.variable_scope(f'block-up-{len(residuals) - i}'):
            print(f'{len(residuals) - i}: {net.shape[1:]} -> {residual.shape[1:]}')
            net = conv2d_transpose(net, residual.shape[-1], kernel_size=3, stride=2, normalizer_fn=batch_norm)
            net = conv2d(net, residual.shape[-1], kernel_size=1, normalizer_fn=batch_norm)
            net = tf.image.resize_image_with_crop_or_pad(net, *residual.shape.as_list()[1:3])
            net += residual
    
    net = conv2d(net, 3, kernel_size=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
    return net