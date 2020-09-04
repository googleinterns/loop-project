import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D, DepthwiseConv2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops


class WeightedConv2D(Conv2D):
  """Weighted 2D convolution layer class.

  A convolution layer that uses a linear combination of template weights
  instead of learnable filter weights.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    templates: None or a tuple of two tensors (T_k, T_b), where T_k stores
      weights of template kernels, T_b represents template biases.T_k should
      have shape (N, kernel_size, kernel_size, input_channel, filters), T_b
      should be of size (N, filters), where `N` is the number of templates,
      `kernel_size` is an integer size of the convolution window,
      `input_channel` is the number of channels in the input tensor and
      `filters` is the number of filters in the convolution.
      If `templates` is None then the templates are initialized with
      `kernel_initializer` and `bias_initializer` and learned.
    num_templates: integer number of templates. If the parameter `templates` is
      not None then `num_templates` specifies the number of templates to learn,
      otherwise `num_templates` parameter is ignored.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

  def __init__(self, templates=None, num_templates=10, **kwargs):
    super(WeightedConv2D, self).__init__(**kwargs)

    if templates is None:
      self.templates_learnable = True
      self.num_templates = num_templates
    elif len(templates) != 2:
      raise ValueError("templates parameter should be a tuple of two or None.")
    else:
      self.templates_learnable = False
      self.template_kernel = tf.cast(templates[0], dtype=tf.float32)
      self.template_bias = tf.cast(templates[1], dtype=tf.float32)
      self.num_templates = self.template_kernel.shape[0]

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    self.kernel_shape = self.kernel_size + (input_channel, self.filters)
    template_kernel_shape = (self.num_templates, *self.kernel_shape)
    template_bias_shape = (self.num_templates, self.filters)

    if self.templates_learnable:
      # allocate new weights for the leanable templates
      self.template_kernel = self.add_weight(
          name="template_kernel",
          shape=template_kernel_shape,
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          trainable=True,
          dtype=self.dtype)
      if self.use_bias:
        self.template_bias = self.add_weight(
            name="template_bias",
            shape=template_bias_shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
      else:
        self.template_bias = None
        self.bias = None
    else:
      self.kernel = tf.constant(np.zeros(self.kernel_shape), dtype=tf.float32)
      self.kernel = tf.constant(np.zeros(self.filters), dtype=tf.float32)
      # check if the templates tensors shape is correct w.r.t. input shape and
      # parameters
      if not np.all(self.template_kernel.shape == template_kernel_shape):
        raise ValueError("The shape of given template kernel must be " +
                         str(template_kernel_shape) + ", got " +
                         str(self.template_kernel.numpy().shape))
      if not np.all(self.template_bias.shape == template_bias_shape):
        raise ValueError("The shape of given template bias must be " +
                         str(template_bias_shape) + ", got " +
                         str(self.template_bias.numpy().shape))

    channel_axis = self._get_channel_axis()
    layer_input_spec = InputSpec(ndim=self.rank + 2,
                                 axes={channel_axis: input_channel})
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (layer_input_spec, mixture_input_spec)

    self._build_conv_op_input_shape = input_shape
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    self._conv_op_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=tensor_shape.TensorShape(self.kernel_shape),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)
    self.built = True

  def call(self, inputs):
    """Computes convolution of the layer input with the weighted sum of the
    template kernel weights.

    Arguments:
    inputs: a tuple of two tensors: (layer_inputs, mixture_weights).
      layer_inputs: layer input tensor.
      mixture_weights: a tensor of shape (num_templates,)
      representing the weights of each template in the sum."""

    layer_inputs, mixture_weights = inputs
    mixture_shape = mixture_weights.shape
    if not (len(mixture_shape) == 1 and mixture_shape[0] == self.num_templates):
      raise ValueError("The shape of mixture_weights should be ",
                       str((self.num_templates,)), ", got ", str(mixture_shape))

    reshaped_mix_w = tf.reshape(mixture_weights,
                                [self.num_templates, 1, 1, 1, 1])
    self.kernel = tf.reduce_sum(reshaped_mix_w * self.template_kernel, axis=0)
    if self.use_bias:
      reshaped_mix_w = tf.reshape(mixture_weights, [self.num_templates, 1])
      self.bias = tf.reduce_sum(reshaped_mix_w * self.template_bias, axis=0)

    return super(WeightedConv2D, self).call(layer_inputs)

  def get_config(self):
    config = super(WeightedConv2D, self).get_config()
    config["templates_learnable"] = self.templates_learnable
    return config


class WeightedDepthwiseConv2D(DepthwiseConv2D):
  """ Weighted depthwise separable convolution.

  Performs a linear combination of depthwise convolutions with predefined
  kernels (aka templates).

  Arguments:
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    templates: None or a tuple of two tensors (T_k, T_b), where T_k stores
      weights of template kernels, T_b represents template biases.T_k should
      have shape (N, kernel_size, kernel_size, input_channel, depth_multiplier),
      T_b should be of size (N, depth_multiplier*input_channel), where `N` is
      the number of templates, `kernel_size` is an integer size of the
      convolution window, `input_channel` is the number of channels in the input
      tensor and `depth_multiplier` is the number of output channels for each
      layer input channel.
      If `templates` is None then the templates are initialized with
      `depthwise_initializer` and `bias_initializer` and learned.
    num_templates: integer number of templates. If the parameter `templates` is
      not None then `num_templates` specifies the number of templates to learn,
      otherwise `num_templates` parameter is ignored.:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be 'channels_last'.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its 'activation') (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`)."""

  def __init__(self, templates=None, num_templates=10, **kwargs):
    super(WeightedDepthwiseConv2D, self).__init__(**kwargs)
    if templates is None:
      self.templates_learnable = True
      self.num_templates = num_templates
    elif len(templates) != 2:
      raise ValueError("templates parameter should be a tuple of two or None.")
    else:
      self.templates_learnable = False
      self.template_kernel = tf.cast(templates[0], dtype=tf.float32)
      self.template_bias = tf.cast(templates[1], dtype=tf.float32)
      self.num_templates = self.template_kernel.shape[0]

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError("Inputs to `DepthwiseConv2D` should have rank 4. "
                       "Received input shape:", str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError("The channel dimension of the inputs to "
                       "`DepthwiseConv2D` "
                       "should be defined. Found `None`.")
    input_dim = int(input_shape[channel_axis])
    self.kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                         input_dim, self.depth_multiplier)
    template_kernel_shape = (self.num_templates, *self.kernel_shape)
    template_bias_shape = (self.num_templates,
                           input_dim * self.depth_multiplier)

    if self.templates_learnable:
      # allocating new template kernels
      self.template_kernel = self.add_weight(
          shape=template_kernel_shape,
          initializer=self.depthwise_initializer,
          name="depthwise_template_kernel",
          regularizer=self.depthwise_regularizer,
          constraint=self.depthwise_constraint)
      if self.use_bias:
        self.template_bias = self.add_weight(shape=template_bias_shape,
                                             initializer=self.bias_initializer,
                                             name="template_bias",
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
      else:
        self.bias = None
    else:
      self.depthwise_kernel = tf.constant(np.zeros(self.kernel_shape),
                                          dtype=tf.float32)
      self.bias = tf.constant(np.zeros(input_dim * self.depth_multiplier),
                              dtype=tf.float32)
      # check if the templates tensors shape is correct w.r.t. input shape and
      # parameters
      if not np.all(self.template_kernel.shape == template_kernel_shape):
        raise ValueError("The shape of given template kernel must be " +
                         str(template_kernel_shape) + ", got " +
                         str(self.template_kernel.numpy().shape))

      if not np.all(self.template_bias.shape == template_bias_shape):
        raise ValueError("The shape of given template bias must be " +
                         str(template_bias_shape) + ", got " +
                         str(self.template_bias.numpy().shape))
    # Set input spec.
    layer_input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (layer_input_spec, mixture_input_spec)
    self.built = True

  def call(self, inputs):
    """Computes depthwise convolution of the layer input with the linear
    combination of the template kernel weights.

    Arguments:
    inputs: a tuple of two tensors: (layer_inputs, mixture_weights).
      layer_inputs: layer input tensor.
      mixture_weights: a tensor of shape (num_templates,)
      representing the weights of each template in the sum."""

    layer_inputs, mixture_weights = inputs
    mixture_shape = mixture_weights.shape
    if not (len(mixture_shape) == 1 and mixture_shape[0] == self.num_templates):
      raise ValueError("The shape of mixture_weights should be ",
                       str((self.num_templates,)), ", got ", str(mixture_shape))

    reshaped_mix_w = tf.reshape(mixture_weights,
                                [self.num_templates, 1, 1, 1, 1])
    self.depthwise_kernel = tf.reduce_sum(reshaped_mix_w * self.template_kernel,
                                          axis=0)
    if self.use_bias:
      reshaped_mix_w = tf.reshape(mixture_weights, [self.num_templates, 1])
      self.bias = tf.reduce_sum(reshaped_mix_w * self.template_bias, axis=0)

    return super(WeightedDepthwiseConv2D, self).call(layer_inputs)

  def get_config(self):
    config = super(WeightedDepthwiseConv2D, self).get_config()
    config["num_templates"] = self.num_templates
    config["templates_learnable"] = self.templates_learnable
    return config
