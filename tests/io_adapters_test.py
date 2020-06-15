import tensorflow as tf
from tensorflow import keras
import numpy as np
import unittest

from lib import io_adapter as adp



class InputAdapterTest(tf.test.TestCase):
  def setUp(self):
    super(InputAdapterTest, self).setUp()
    self.default_size = 32
    self.default_depth = 64

  # tests if the output of the adaptes is of correct shape
  def test_output_shape(self):
    input_shape = (64, 64, 3)
    batch_size = 16
    expected_out_shape = (batch_size, self.default_size,
                          self.default_size, self.default_depth)
    adapter = self._create_default_adapter(input_shape)
    out = adapter(np.zeros((batch_size, *input_shape)))
    self.assertShapeEqual(np.zeros(expected_out_shape), out)

  def test_small_out_size(self):
    input_shape = (128, 128, 8)
    with self.assertRaises(Exception):
      self._create_default_adapter(input_shape)

  def test_non_divisible(self):
    input_shape = (50, 50, 3)
    with self.assertRaises(Exception):
     self. _create_default_adapter(input_shape)

  def _create_default_adapter(self, input_shape):
    adapter = adp.create_input_adapter(input_shape,
                              size=self.default_size,
                              depth=self.default_depth)
    return adapter



class OutputAdapterTest(tf.test.TestCase):

  def setUp(self):
    super(OutputAdapterTest, self).setUp()

  def test_out_shape(self):
    input_shape = (32, 32, 40)
    batch = 32
    input_tensor = tf.random.normal([batch, *input_shape])
    block_size = 4
    out_adapter = adp.create_output_adapter(
                        input_shape, block_size=block_size)
    out = out_adapter(input_tensor)
    expected_num_c = input_shape[2]*block_size*block_size
    expected_out_shape = (batch, 1, 1, expected_num_c)
    self.assertAllEqual(expected_out_shape, out.shape)

  def test_bad_block_size(self):
    input_shape = (32, 32, 40)
    with self.assertRaises(ValueError):
      out_adapter = adp.create_output_adapter(
                        input_shape, block_size=3.5)

  def test_bad_pool_stride(self):
    input_shape = (32, 32, 40)
    with self.assertRaises(ValueError):
      out_adapter = adp.create_output_adapter(
                        input_shape, pool_stride = '3')

  def test_bad_input_shape(self):
    input_shape = (32, 32)
    with self.assertRaises(ValueError):
      out_adapter = adp.create_output_adapter(
                        input_shape, block_size=4)
