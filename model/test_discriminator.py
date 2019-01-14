import tensorflow as tf
import unittest
from .discriminator import build_discriminator

class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_build_discriminator(self):
        input_tensor = tf.placeholder(tf.float32, shape=(None, 218, 178, 3))
        output_tensor = build_discriminator(input_tensor, layer_num=6)

        self.assertIsNotNone(output_tensor)
        self.assertEqual([None, 1], output_tensor.shape.as_list())


if __name__ == "__main__":
    unittest.main()