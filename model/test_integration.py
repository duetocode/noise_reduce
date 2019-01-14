import tensorflow as tf
import unittest
from .generator import build_generator
from .discriminator import build_discriminator

class TestIntegration(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_integration(self):
        input_tensor = tf.placeholder(tf.float32, shape=[None, 218, 178, 3])
        fake_images = build_generator(input_tensor)
        fake_logits = build_discriminator(fake_images)

        self.assertIsNotNone(fake_images)
        self.assertIsNotNone(fake_logits)

        self.assertEqual([None, 218, 178, 3], fake_images.shape.as_list())
        self.assertEqual([None, 1], fake_logits.shape.as_list())


if __name__ == "__main__":
    unittest.main()