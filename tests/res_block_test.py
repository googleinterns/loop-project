import unittest
import tensorflow as tf
from lib import res_block


class BlockTest(tf.test.TestCase):
  """ ResBlock test class. """

  def _run_standard_block(self, input_tensor):
    """creates and runs a ResBlock with standard parameters."""
    block = res_block.ResBlock(kernel_size=3,
                               expansion_factor=6,
                               activation='relu')
    block.build(tf.shape(input_tensor))
    return block(input_tensor)

  def test_basic(self):
    """Checking if the input and output tensors shapes match."""
    input_shape = (32, 128, 128, 64)
    input_val = tf.random.normal([*input_shape])
    out = self._run_standard_block(input_val)
    self.assertShapeEqual(input_val.numpy(), out)

  def test_expansion_wrong_val(self):
    """checks if the wrong expansion_factor value raises an error."""
    with self.assertRaises(ValueError):
      res_block.ResBlock(kernel_size=3,
                         expansion_factor=0,
                         activation='relu')

  def test_zeros_input(self):
    """tests if the result of passing a zero tensor is also zeros."""
    input_shape = (8, 16, 16, 40)
    input_val = tf.zeros([*input_shape])
    out = self._run_standard_block(input_val)
    self.assertAllEqual(input_val, out)

  def test_wrong_activation(self):
    """checks if passing wrong activation results in an error."""
    with self.assertRaises(Exception):
      res_block.ResBlock(kernel_size=3,
                         expansion_factor=6,
                         activation='sigmoid')

if __name__ == '__main__':
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
