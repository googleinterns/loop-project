import tensorflow as tf
from tensorflow.python.keras import regularizers
from lib import weighted_layers_v2 as wl

class ResBlockTemplate():
  """Custom residual block template class.

  Stores the templates of weighted expansion, depthwise convolution and
  projection layers of the custom weighted residual block.

  Arguments:
  expansion_template: None or a tuple of two expansion layer
    (1x1 WeightedConv2D) templates tensors (T_k, T_b), where T_k stores weights
    of template kernels, T_b represents template biases. T_k should have shape
    (N, kernel_size, kernel_size, input_channel, filters), T_b should be of
    shape (N, filters), where `N` is the number of templates, `kernel_size` is
    an integer size of the convolution window, `input_channel` is the number of
    channels in the input tensor and `filters` is the number of filters in the
    convolution.
  depthwise_template: None or a tuple of two tensors (T_k, T_b), where T_k
    stores weights of template kernels, T_b represents template biases of the
    weighted depthwise convolution layer. T_k should have shape
    (N, kernel_size, kernel_size, input_channel, depth_multiplier), T_b should
    be of size (N, depth_multiplier*input_channel), where `N` is the number of
    templates, `kernel_size` is an integer size of the convolution window,
    `input_channel` is the number of channels in the input tensor and
    `depth_multiplier` is the number of output channels for each layer input
    channel.
  projection_template: None or a tuple of two projection layer
    (1x1 WeightedConv2D) templates tensors (T_k, T_b), where T_k stores weights
    of template kernels, T_b represents template biases. T_k should have shape
    (N, kernel_size, kernel_size, input_channel, filters), T_b should be of
    shape (N, filters), where `N` is the number of templates, `kernel_size` is
    an integer size of the convolution window, `input_channel` is the number of
    channels in the input tensor and `filters` is the number of filters in the
    convolution.
  bn_1_template: None or a tuple of four templates tensors (gamma, beta) for
    the first batch normalization layer. All templates tensors should be of
    size (N, n_channels), where N is the number of templates, n_channels is
    the number of channels of the layer inputs.
  bn_2_template: None or a tuple of four templates tensors (gamma, beta) for
    the second batch normalization layer. All templates tensors should be of
    size (N, n_channels), where N is the number of templates, n_channels is
    the number of channels of the layer inputs.
  bn_3_template: None or a tuple of four templates tensors (gamma, beta) for
    the third batch normalization layer. All templates tensors should be of
    size (N, n_channels), where N is the number of templates, n_channels is
    the number of channels of the layer inputs.
  """
  def __init__(self, expansion_template=None, depthwise_template=None,
               projection_template=None, bn_1_template=None,
               bn_2_template=None, bn_3_template=None):
    self._check_dims(expansion_template, depthwise_template,
                     projection_template)

    self.expansion_template = expansion_template
    self.depthwise_template = depthwise_template
    self.projection_template = projection_template
    self.bn_1_template = bn_1_template
    self.bn_2_template = bn_2_template
    self.bn_3_template = bn_3_template

  def get_expansion_template(self):
    """returns the expansion template."""
    if self.expansion_template is None:
      return "he_normal", "zeros"
    kernel_init = tf.constant_initializer(self.expansion_template[0])
    bias_init = tf.constant_initializer(self.expansion_template[1])
    return kernel_init, bias_init

  def get_bn1_template(self):
    """returns the first batch normalization template."""
    if self.bn_1_template is None:
      return "zeros", "ones"
    beta_init = tf.constant_initializer(self.bn_1_template[0])
    gamma_init = tf.constant_initializer(self.bn_1_template[1])
    return gamma_init, beta_init

  def get_bn2_template(self):
    """returns the first batch normalization template."""
    if self.bn_2_template is None:
      return "zeros", "ones"
    beta_init = tf.constant_initializer(self.bn_2_template[0])
    gamma_init = tf.constant_initializer(self.bn_2_template[1])
    return gamma_init, beta_init

  def get_bn3_template(self):
    """returns the first batch normalization template."""
    if self.bn_3_template is None:
      return "zeros", "ones"
    beta_init = tf.constant_initializer(self.bn_3_template[0])
    gamma_init = tf.constant_initializer(self.bn_3_template[1])
    return gamma_init, beta_init

  def get_depthwise_template(self):
    """returns the depthwise convolution template."""
    if self.depthwise_template is None:
      return "he_normal", "zeros"
    kernel_init = tf.constant_initializer(self.depthwise_template[0])
    bias_init = tf.constant_initializer(self.depthwise_template[1])
    return kernel_init, bias_init

  def get_projection_template(self):
    """returns the projection template."""
    if self.projection_template is None:
      return "he_normal", "zeros"
    kernel_init = tf.constant_initializer(self.projection_template[0])
    bias_init = tf.constant_initializer(self.projection_template[1])
    return kernel_init, bias_init

  def set_expansion_template(self, expansion_template):
    """sets the expansion template."""
    self._check_dims(expansion_template, None, None)
    self.expansion_template = expansion_template

  def set_depthwise_template(self, depthwise_template):
    """sets the depthwise convolution template."""
    self._check_dims(None, depthwise_template, None)
    self.depthwise_template = depthwise_template

  def set_projection_template(self, projection_template):
    """sets the projection template."""
    self._check_dims(None, None, projection_template)
    self.projection_template = projection_template

  def _check_dims(self, expansion, depthwise, projection):
    """Checks if the templates are tensors of correct dimensionality."""
    if expansion is not None:
      if (not (isinstance(expansion[0], tf.Tensor) or
               isinstance(expansion[0], tf.Variable)) or
          expansion[0].get_shape().ndims != 5):
        raise ValueError(
            "Expansion kernel template should be 5-dimensional tensor.")
      if (not (isinstance(expansion[1], tf.Tensor) or
               isinstance(expansion[1], tf.Variable)) or
          expansion[1].get_shape().ndims != 2):
        raise ValueError(
            "Expansion bias template should be 2-dimensional tensor.")

    if depthwise is not None:
      if (not (isinstance(depthwise[0], tf.Tensor) or
               isinstance(depthwise[0], tf.Variable)) or
          depthwise[0].get_shape().ndims != 5):
        raise ValueError(
            "Depthwise kernel template should be 5-dimensional tensor.")
      if (not (isinstance(depthwise[1], tf.Tensor) or
               isinstance(depthwise[1], tf.Variable)) or
          depthwise[1].get_shape().ndims != 2):
        raise ValueError(
            "Depthwise bias template should be 2-dimensional tensor.")

    if projection is not None:
      if (not (isinstance(projection[0], tf.Tensor) or
               isinstance(projection[0], tf.Variable)) or
          projection[0].get_shape().ndims != 5):
        raise ValueError(
            "Projection kernel template should be 5-dimensional tensor.")
      if (not (isinstance(projection[1], tf.Tensor) or
               isinstance(projection[1], tf.Variable)) or
          projection[1].get_shape().ndims != 2):
        raise ValueError(
            "Projection bias template should be 2-dimensional tensor.")


class WeightedResBlockSeparateBN(tf.keras.layers.Layer):
  """ Weighted ResBlock with separate batch normalization.

  Arguments:
  resblock: WeightedResBlock object.
  """
  def __init__(self, resblock, **kwargs):
    super(WeightedResBlockSeparateBN, self).__init__(**kwargs)
    if not isinstance(resblock, WeightedResBlock):
      raise TypeError("resblock should be a WeightedResBlock class instance.")

    self.resblock = resblock
    self.bn1 = tf.keras.layers.BatchNormalization(
        center=False, scale=False)
    self.bn2 = tf.keras.layers.BatchNormalization(
        center=False, scale=False)
    self.bn3 = tf.keras.layers.BatchNormalization(
        center=False, scale=False)

  def build(self, input_shape):
    super(WeightedResBlockSeparateBN, self).build(input_shape)
    self.resblock.build(input_shape)

  def call(self, inputs, training=None):
    """Calls regular batch normalization between resblock operations.
    """
    layer_input, mix_weights = inputs
    x = self.resblock.conv1([layer_input, mix_weights])
    x = self.bn1(x, training)
    x = self.resblock.bn1([x, mix_weights], training)
    x = self.resblock.activation(x)

    x = self.resblock.conv2([x, mix_weights])
    x = self.bn2(x, training)
    x = self.resblock.bn2([x, mix_weights], training)
    x = self.resblock.activation(x)

    x = self.resblock.conv3([x, mix_weights])
    x = self.bn3(x, training)
    x = self.resblock.bn3([x, mix_weights], training)
    x += layer_input
    return x

  def get_config(self):
    return super(WeightedResBlockSeparateBN, self).get_config()



class WeightedResBlock(tf.keras.Model):
  """ A ResBlock module class with expansion, depthwise convolution and
  projection that uses weighted convolutions.

  In this ResBlock, standard 2D convolutions are replaced by 1x1 weighted
  convolution that expands the input tensor along the channel dimension,
  weighted depthwise convolution and weighted 1x1 convolution that projects the
  tensor back to the original number of channels.

   Args:
     kernel_size: size of the depthwise convolution kernel.
     expansion_factor: expansion factor of the first 1x1 convolution.
      e.g., if the input tensor has N channels, then the first 1x1
      convolution layer will expand it to expansion_factor*N channels.
      activation: activation name or function. Supported function
      names are 'relu', 'relu6', 'lrelu', 'swish'.
      template: a ResBlockTemplate object.
      kernel_reg: kernel regularizer parameter.
      """

  def __init__(self, kernel_size=3, expansion_factor=6, activation="relu",
               num_templates=10, template=None, kernel_reg=1e-5, **kwargs):
    super(WeightedResBlock, self).__init__(**kwargs)
    if expansion_factor < 1:
      raise ValueError("The expansion factor value should be "
                       "greater than or equal to one.")

    self.expansion_factor = expansion_factor
    self.activation = self.map_activation_fn(activation)
    self.kernel_size = kernel_size
    self.template = ResBlockTemplate() if template is None else template
    self.num_templates = num_templates
    self.kernel_reg = kernel_reg

  def build(self, input_shape):
    input_channel = input_shape[0][-1]
    self.expanded_channel = input_channel * self.expansion_factor

    kernel_init, bias_init = self.template.get_expansion_template()
    self.conv1 = wl.WeightedConv2D(
        filters=self.expanded_channel, kernel_size=1, strides=(1, 1),
        padding="same", num_templates=self.num_templates,
        kernel_initializer=kernel_init,
        kernel_regularizer=regularizers.l2(self.kernel_reg),
        bias_initializer=bias_init)
    self.conv1.build(input_shape)

    beta, gamma = self.template.get_bn1_template()
    self.bn1 = wl.WeightedBatchNormalizationSeparate(
        num_templates=self.num_templates, gamma_initializer=gamma,
        beta_initializer=beta)

    depthwise_kernel_init, bias_init = self.template.get_depthwise_template()
    self.conv2 = wl.WeightedDepthwiseConv2D(
        kernel_size=self.kernel_size, strides=(1, 1), padding="same",
        num_templates=self.num_templates,
        depthwise_initializer=depthwise_kernel_init,
        bias_initializer=bias_init)

    cov2_in_shape = ((input_shape[0][0], input_shape[0][1], input_shape[0][2],
                      self.expanded_channel), (self.num_templates,))
    self.conv2.build(cov2_in_shape)

    beta, gamma = self.template.get_bn2_template()
    self.bn2 = wl.WeightedBatchNormalizationSeparate(
        num_templates=self.num_templates, gamma_initializer=gamma,
        beta_initializer=beta)

    kernel_init, bias_init = self.template.get_projection_template()
    self.conv3 = wl.WeightedConv2D(
        filters=input_channel, kernel_size=1, strides=(1, 1), padding="same",
        num_templates=self.num_templates,
        kernel_initializer=kernel_init,
        kernel_regularizer=regularizers.l2(self.kernel_reg),
        bias_initializer=bias_init)
    self.conv3.build(cov2_in_shape)

    beta, gamma = self.template.get_bn3_template()
    self.bn3 = wl.WeightedBatchNormalizationSeparate(
        num_templates=self.num_templates, gamma_initializer=gamma,
        beta_initializer=beta)

    self.built = True

  def call(self, inputs, training=None):
    layer_input, mix_weights = inputs
    x = self.conv1([layer_input, mix_weights])
    x = self.bn1([x, mix_weights], training)
    x = self.activation(x)

    x = self.conv2([x, mix_weights])
    x = self.bn2([x, mix_weights], training)
    x = self.activation(x)

    x = self.conv3([x, mix_weights])
    x = self.bn3([x, mix_weights], training)
    x += layer_input
    return x

  def map_activation_fn(self, activation):
    """Maps activation function name to function."""
    if callable(activation):
      return activation

    switcher = {"relu": tf.nn.relu,
                "relu6": tf.nn.relu6,
                "lrelu": tf.nn.leaky_relu,
                "swish": tf.nn.swish}

    res = switcher.get(activation)
    if not res:
      raise Exception("Given activation function is not supported.")
    return res

  def _get_input_channel(self, input_shape):
    if input_shape.dims[-1].value is None:
      raise ValueError("The channel dimension of the inputs "
                       "should be defined. Found `None`.")
    return int(input_shape[-1])

class MixtureWeight(tf.keras.layers.Layer):
  """Mixture weights layer.

  Arguments:
  num_templates: integer number of templates in weighted block.
  initializer: mixture weights initializer (see `keras.initializers`).
  regularizer: mixture weights regularizer (see `keras.regularizers`).
  constraint: constraint function applied to mixture weight vector
    (see `keras.constraints`)
  dtype: type of variable."""
  def __init__(self, num_templates=10, initializer="glorot_uniform",
               regularizer=None, constraint=None, dtype=tf.float32,
               **kwargs):
    super(MixtureWeight, self).__init__(**kwargs)
    self.num_templates = num_templates
    self.initializer = initializer
    self.regularizer = regularizer
    self.constraint = constraint
    self.tensor_type = dtype

  def build(self, input_shape):
    self.mixture_weights = self.add_weight(
        name="mixture_weight",
        shape=(1, self.num_templates),
        initializer=self.initializer,
        regularizer=self.regularizer,
        constraint=self.constraint,
        trainable=True,
        dtype=self.tensor_type)
    self.built = True

  def call(self, inputs):
    return tf.nn.softmax(self.mixture_weights, axis=1)

  def get_config(self):
    config = super(MixtureWeight, self).get_config()
    config.update({"num_templates": self.num_templates})
    return config
