import tensorflow as tf

def least_loss(fake_logits, real_logits):
    
  G_loss = tf.reduce_mean((fake_logits - 1)**2) / 2
  
  D_loss_fake_part = tf.reduce_mean(fake_logits**2)
  D_loss_real_part = tf.reduce_mean((real_logits - 1)**2)
  
  D_loss = (D_loss_fake_part + D_loss_real_part) / 2
  
  return G_loss, D_loss