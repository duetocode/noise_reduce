import tensorflow as tf

def least_loss(fake_logits, real_logits):
    
  G_loss = tf.reduce_mean((fake_logits - 1)**2) / 2
  
  D_loss_fake_part = tf.reduce_mean(fake_logits**2)
  D_loss_real_part = tf.reduce_mean((real_logits - 1)**2)
  
  D_loss = (D_loss_fake_part + D_loss_real_part) / 2
  
  return G_loss, D_loss


def wasserstein_gp(fake_logits, real_logits, fake_images, real_images, batch_size, discriminator, lambda_GP=10.0):
  G_loss = -tf.reduce_mean(fake_logits)
  D_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

  alpha = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
  alpha = tf.expand_dims(alpha, axis=-1)
  alpha = tf.expand_dims(alpha, axis=-1)

  difference = real_images - fake_images
  interpolates = real_images + (alpha * difference)

  gradients = tf.gradients(discriminator(interpolates), interpolates)[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
  gradient_penalty = tf.reduce_mean((slopes-1.)**2)
  D_loss += lambda_GP * gradient_penalty

  return G_loss, D_loss