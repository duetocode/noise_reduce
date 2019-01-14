import unittest
import tensorflow as tf
from .generator import build_generator

class TestGenerator(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_build_generator(self):
        input_tensor = tf.placeholder(tf.float32, shape=[None, 218, 178, 3])
        output = build_generator(input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual([None, 218, 178, 3], output.shape.as_list())


if __name__ == '__main__':
    unittest.main()