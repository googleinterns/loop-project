import unittest
import numpy as np
import tensorflow as tf
from lib.weighted_layers_v2 import *
from lib.weighted_resblock import MixtureWeight

class WeightedConv2DTest(tf.test.TestCase):
  """WeightedConv2D test class."""

  def setUp(self):
    """Sets default parameters."""
    super(WeightedConv2DTest, self).setUp()
    self.kernel_size = 3
    self.activation = 'relu'
    self.filters = 40
    self.input_channel = 20
    self.num_templates = 10
    self.kernel = np.random.rand(self.num_templates, self.kernel_size,
                                 self.kernel_size, self.input_channel,
                                 self.filters)
    self.bias = np.random.rand(self.num_templates, self.filters)
    self.kernel_init = tf.constant_initializer(self.kernel)
    self.bias_init = tf.constant_initializer(self.bias)
    self.padding = 'same'
    xi_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    self.xi = MixtureWeight(num_templates=self.num_templates,
                            initializer=xi_init)

  def _create_default_w_conv(self):
    """Creates an instance of WeightedConv2D with dedault parameters."""
    return WeightedConv2D(filters=self.filters, activation=self.activation,
                          padding=self.padding, kernel_size=self.kernel_size,
                          num_templates=self.num_templates,
                          kernel_initializer=self.kernel_init,
                          bias_initializer=self.bias_init)

  def _get_default_inputs(self, in_shape):
    """returns default layer inputs."""
    layer_inputs = tf.Variable(np.random.rand(*in_shape), dtype=tf.float32)

    return [layer_inputs, self.xi(None)]

  def test_output_shape(self):
    """checks if the shape of the output tensor is correct."""
    w_conv = self._create_default_w_conv()
    input_shape = (32, 16, 16, self.input_channel)
    inputs = self._get_default_inputs(input_shape)
    output = w_conv(inputs)
    expected_shape = (32, 16, 16, self.filters)
    self.assertAllEqual(expected_shape, output.shape)


  def test_output_values(self):
    """checks if the output tensor is computed correctly."""
    w_conv = self._create_default_w_conv()
    w_conv.activation = None
    input_shape = (32, 16, 16, self.input_channel)
    inputs = self._get_default_inputs(input_shape)
    w_output = w_conv(inputs)

    # the output of weighted convolution should be same as linear combination of
    # outputs of regular convolution with template weights in case when no
    # activation is used.
    expected_output = tf.zeros_like(w_output)
    conv = tf.keras.layers.Conv2D(filters=self.filters, activation=None,
                                  padding=self.padding,
                                  kernel_size=self.kernel_size)
    conv.build(input_shape)
    for t in range(self.num_templates):
      conv.kernel = self.kernel[t]
      conv.bias = self.bias[t]
      conv_out = conv(inputs[0])
      expected_output += inputs[1][t]*conv_out
    self.assertAllClose(expected_output, w_output, rtol=1e-05)


class WeightedDepthwiseConv2DTest(tf.test.TestCase):
  """Weighted depthwise convolution test class."""

  def setUp(self):
    """Sets default parameters."""
    super(WeightedDepthwiseConv2DTest, self).setUp()
    self.kernel_size = 3
    self.activation = 'relu'
    self.depth_multiplier = 2
    self.input_channel = 20
    self.num_templates = 10
    self.kernel = np.random.rand(self.num_templates, self.kernel_size,
                                 self.kernel_size, self.input_channel,
                                 self.depth_multiplier).astype(np.float32)
    self.bias = np.random.rand(
        self.num_templates,
        self.input_channel * self.depth_multiplier).astype(np.float32)

    self.kernel_init = tf.constant_initializer(self.kernel)
    self.bias_init = tf.constant_initializer(self.bias)
    self.padding = 'same'
    self.xi_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    self.xi = MixtureWeight(num_templates=self.num_templates,
                            initializer=self.xi_initializer)

  def _create_default_depth_conv(self):
    """Creates a WeightedDepthwiseConv2D instance with default parameters."""
    return WeightedDepthwiseConv2D(
        depth_multiplier=self.depth_multiplier, activation=self.activation,
        padding=self.padding, kernel_size=self.kernel_size,
        num_templates=self.num_templates,
        depthwise_initializer=self.kernel_init,
        bias_initializer=self.bias_init)

  def _get_default_inputs(self, in_shape):
    """returns default layer inputs."""
    layer_inputs = tf.Variable(np.random.rand(*in_shape), dtype=tf.float32)
    return [layer_inputs, self.xi(None)]

  def test_output_shape(self):
    """checks if the shape of the output tensor is correct."""
    w_d_conv = self._create_default_depth_conv()
    input_shape = (32, 64, 64, self.input_channel)
    inputs = self._get_default_inputs(input_shape)
    output = w_d_conv(inputs)
    expected_shape = (32, 64, 64, self.input_channel*self.depth_multiplier)
    self.assertAllEqual(expected_shape, output.shape)

  def test_output_value(self):
    """checks if the value of the output tensor is correct."""
    w_d_conv = self._create_default_depth_conv()
    w_d_conv.activation = None
    input_shape = (32, 16, 16, self.input_channel)
    inputs = self._get_default_inputs(input_shape)
    w_d_output = w_d_conv(inputs)

    # the output of weighted convolution should be same as linear combination of
    # outputs of regular convolution with template weights in case when no
    # activation is used.
    expected_output = tf.zeros_like(w_d_output)
    conv = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=self.depth_multiplier, activation=None,
        padding=self.padding, kernel_size=self.kernel_size)
    conv.build(input_shape)
    for t in range(self.num_templates):
      conv.depthwise_kernel = self.kernel[t]
      conv.bias = self.bias[t]
      conv_out = conv(inputs[0])
      expected_output += inputs[1][t]*conv_out
    self.assertAllClose(expected_output, w_d_output, rtol=1e-05)

class WeightedBatchNormalizationTest(tf.test.TestCase):
  """"WeightedBatchNormalizationSeparate test class."""
  def setUp(self):
    """Sets default parameters."""
    self.num_templates = 10
    self.input_channels = 40
    self.gamma_template = np.random.rand(self.num_templates,
                                         self.input_channels).astype(np.float32)
    self.beta_template = np.random.rand(self.num_templates,
                                        self.input_channels).astype(np.float32)
    self.beta_init = tf.constant_initializer(self.beta_template)
    self.gamma_init = tf.constant_initializer(self.gamma_template)
    self.xi_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    self.xi = MixtureWeight(num_templates=self.num_templates,
                            initializer=self.xi_initializer)

  def test_output_shape(self):
    """checks if the output shape is same as input shape."""
    input_shape = (256, 16, 16, self.input_channels)
    inputs = tf.random.normal(input_shape)
    bn = WeightedBatchNormalizationSeparate(num_templates=self.num_templates,
                                            gamma_initializer=self.gamma_init,
                                            beta_initializer=self.beta_init)
    outputs = bn([inputs, self.xi(None)], training=True)
    self.assertAllEqual(input_shape, outputs.shape)

  def test_output_moments(self):
    """checks if the moments of the output tensor match to the value of mixture
    of moments."""
    input_shape = (256, 16, 16, self.input_channels)
    inputs = tf.random.normal(input_shape, mean=2.5, stddev=8.0)
    bn = WeightedBatchNormalizationSeparate(num_templates=self.num_templates,
                                            gamma_initializer=self.gamma_init,
                                            beta_initializer=self.beta_init)
    outputs = bn([inputs, self.xi(None)], training=True)
    reduction_axes = [i for i in range(len(input_shape) - 1)]
    mean, var = nn.moments(outputs, reduction_axes)
    reshaped_mix_w = tf.reshape(self.xi(None), [self.num_templates, 1])
    mix_gamma = tf.reduce_sum(reshaped_mix_w * self.gamma_template, axis=0)
    mix_beta = tf.reduce_sum(reshaped_mix_w * self.beta_template, axis=0)
    self.assertAllClose(mean, mix_beta, rtol=1e-03)
    self.assertAllClose(tf.math.sqrt(var), mix_gamma, rtol=1e-03)

if __name__ == '__main__':
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
