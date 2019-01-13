import tensorflow as tf

from model import build_discriminator, build_generator
from loss import least_loss as loss
from dataset import open_new_dataset, wreck

# Hyperparameters
learning_rate = 1e-4
beta1 = 0.5
epochs = 2

# Prepare dataset
dataset = open_new_dataset()
images_iter = dataset.make_initializable_iterator()
images = images_iter.get_next()

wrecked_images = tf.placeholder(tf.float32, shape=(None, 218, 178, 3))
real_images = tf.placeholder(tf.float32, shape=(None, 218, 178, 3))

# Build the graph
with tf.variable_scope('') as scope:
    fake_images = build_generator(wrecked_images)
    fake_logits = build_discriminator(fake_images)
    scope.reuse_variables()
    real_logits = build_discriminator(real_images)

G_loss, D_loss = loss(fake_logits, real_logits)
G_loss = 0.5 * G_loss + 0.5 * tf.losses.absolute_difference(fake_images, real_images)

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

# train

def onEpoch(e, sess):
    sess.run([images_iter.initializer])
    for i in range(10000):
        _real_images = sess.run(images)
        _wrecked_images = wreck(_real_images)
        _, _G_loss = sess.run([G_train_step, G_loss], \
            feed_dict={wrecked_images: _wrecked_images, real_images: _real_images})
        _, _D_loss = sess.run([D_train_step, D_loss], \
            feed_dict={wrecked_images: _wrecked_images, real_images: _real_images})

        print(f'\recho[{e} - ]{i}/10000, D_LOSS:{_D_loss}, G_LOSS:{_G_loss}', end='')
    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(1, epochs+1):
        onEpoch(e, sess)