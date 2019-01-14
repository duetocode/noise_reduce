import tensorflow as tf
import unittest
from tensorflow.contrib.layers import fully_connected
from loss import wasserstein_gp

class TestLosses(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_wasserstein_gp(self):
        batch_size = 3
        fake_images = tf.random_uniform([batch_size, 28, 28, 3], minval=0.0, maxval=1.0)
        real_images = tf.random_uniform([batch_size, 28, 28, 3], minval=0.0, maxval=1.0)
        fake_logits = tf.random_uniform([batch_size, 1], minval=-1.0, maxval=1.0)
        real_logits = tf.random_uniform([batch_size, 1], minval=-1.0, maxval=1.0)

        def discriminator(input_tensor):
            net = tf.reshape(input_tensor, [batch_size, -1])
            net = fully_connected(net, 1)
            return net

        G_loss, D_loss = wasserstein_gp(fake_logits, real_logits, fake_images, real_images, batch_size, discriminator)

        self.assertIsNotNone(G_loss)
        self.assertIsNotNone(D_loss)

if __name__ == "__main__":
    unittest.main()



