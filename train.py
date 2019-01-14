import tensorflow as tf
import PIL
import numpy as np

from PIL import Image
from model import build_discriminator, build_generator
from loss import wasserstein_gp as loss
from dataset import open_new_dataset, wreck

HOME='.'

# Hyperparameters
learning_rate = 1e-4
beta1 = 0.5
epochs = 2
batch_size = 2

iterations_per_epoch = 2500

# Prepare dataset
tf.reset_default_graph()
dataset = open_new_dataset(f'{HOME}/dataset/dataset.tr', \
                           batch_size=batch_size)
images_iter = dataset.make_initializable_iterator()
images = images_iter.get_next()

# Build the graph
wrecked_images = tf.placeholder(tf.float32, shape=(None, 218, 178, 3))
real_images = tf.placeholder(tf.float32, shape=(None, 218, 178, 3))

l1_ratio = 0.3

# Build the graph
with tf.variable_scope('') as scope:
    fake_images = build_generator(wrecked_images)
    fake_logits = build_discriminator(fake_images)
    scope.reuse_variables()
    real_logits = build_discriminator(real_images)
    G_loss, D_loss = loss(fake_logits, real_logits, fake_images, real_images, batch_size, build_discriminator)

G_loss = (1 - l1_ratio) * G_loss + l1_ratio * tf.losses.mean_squared_error(fake_images, real_images)

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

# train
pointer = 0

def save(image, filename):
  img = Image.fromarray(np.uint8((image + 1.0) / 2.0 * 255.0))
  img.save(f'{HOME}/samples/{filename}')  

def log_image(real_images, wreched_images, fake_images):
  global pointer
  pointer += 1
  if pointer > 5: 
    pointer = 1
  
  save(real_images[0], f'{pointer}-real.jpg')
  save(fake_images[0], f'{pointer}-fake.jpg')
  save(wreched_images[0], f'{pointer}-wreched.jpg')
  
saver = tf.train.Saver()


def onEpoch(e, sess):
    sess.run([images_iter.initializer])
    for i in range(iterations_per_epoch):
        _real_images = sess.run(images)
        _wrecked_images = wreck(_real_images)
        _, _, _G_loss, _D_loss, _fake_images = sess.run([G_train_step, D_train_step, G_loss, D_loss, fake_images], \
            feed_dict={wrecked_images: _wrecked_images, real_images: _real_images})

        print(f'\recho[{e} - ]{i}/{iterations_per_epoch}, D_LOSS:{_D_loss}, G_LOSS:{_G_loss}', end='')
        if i % 100 == 0:
          log_image(_real_images, _wrecked_images, _fake_images)
        if i % 500 == 0:
          # persistent weights
          global_steps = e * iterations_per_epoch + i
          saver.save(sess, f'{HOME}/checkpoints/noise_reduce.ckpt', global_step=global_steps)
    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(1, epochs+1):
        onEpoch(e, sess)