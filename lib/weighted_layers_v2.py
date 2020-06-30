import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D, DepthwiseConv2D
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn

class WeightedMixIn():
  """Mix-in class for weighted layers.

  Weighted layers inherit from WeightedMixIn class as well as the corresponding
  layer class to replace trainable weights by mixture of template weights."""

  def __init__(self):
    self.tracked_template_names = {}
    self.templates = {}
    self._base_add_weight = self.add_weight
    self.add_weight = self.add_mix_weight
    self._base_build = self.build
    self.build = self.mixin_build

  def add_template_variable(self, weight_name, field_name=None):
    """adds a variable to be computed as a mixture of templates.

    Arguments:
    weight_name: string name of the variable.
    field_name: string field name of the variable or None. If None then
    field_name is same as weight_name."""

    field_name = weight_name if field_name is None else field_name
    self.tracked_template_names[weight_name] = field_name
    if not hasattr(self, 'add_weight'):
      raise TypeError("The object must have an attribute `add_weight`.")
    # self.add_weight = self.add_mix_weight

  def mixin_build(self, input_shape):
    """replacement for the `build` function of the other parent class.

    Arguments:
    input_shape: shape of layer input."""

    self._base_build(input_shape[0])
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (self.input_spec, mixture_input_spec)

  def add_mix_weight(self, **kwargs):
    """if the variable is assigned to be templated, adds template weights, 
    otherwise adds variable weights as usual."""

    if not hasattr(self, 'num_templates'):
      raise TypeError("The object must have an attribute `num_templates`.")

    if 'name' in kwargs and kwargs['name'] in self.tracked_template_names:
      new_kwargs = kwargs.copy()
      new_kwargs['shape'] = (self.num_templates,) + kwargs['shape']
      self.templates[kwargs['name']] = self._base_add_weight(**new_kwargs)
      return tf.zeros(shape=kwargs['shape'])
    return self._base_add_weight(**kwargs)
  
  def assign_mixture_value(self, name, mixture_weights):
    """computes mixture weights for the variable.

    Arguments:
    name: string variable name.
    mixture_weights: mixture weights tensor."""
    var_ndims = self.templates[name].get_shape().ndims
    new_shape = np.ones(var_ndims)
    new_shape[0] = self.num_templates
    reshaped_mix_w = tf.reshape(mixture_weights, new_shape)
    mixture = tf.reduce_sum(reshaped_mix_w * self.templates[name], axis=0)
    setattr(self, name, mixture)
  
  def assign_all_mixture_values(self, mixture_weights):
    """assigns mixture values to all tracked variables.

    Arguments: 
    mixture_weights: mixture weights tensor."""
    for name in self.tracked_template_names:
      self.assign_mixture_value(name, mixture_weights)

  def reset_value(self, name):
    """sets the value of the variable to None."""
    setattr(self, name, None)

  def reset_all_values(self):
    """sets the value of all tracked variables to None."""
    for name in self.tracked_template_names:
      self.reset_value(name)
    
class WeightedBatchNormalizationSeparate(WeightedMixIn, BatchNormalization):
  """Weighted batch normalization layer.

  This layer performs batch normalizes the activations to have zero mean and
  variance 1 and scales them with parameters gamma and beta, which are computed
  as linear combination of corresponding templates.

  Arguments: 
    num_templates: number of templates.
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.
    trainable: Boolean, if `True` the variables will be marked as trainable.
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random.uniform(shape[-1:], 0.93, 1.07),
          tf.random.uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
  """
  def __init__(self, num_templates=10, **kwargs):
    WeightedMixIn.__init__(self)
    new_kwargs = kwargs.copy()
    new_kwargs["center"] = False
    new_kwargs["scale"] = False
    BatchNormalization.__init__(self, **new_kwargs)
    self.num_templates = num_templates
    if "scale" not in kwargs or kwargs["scale"] == True:
      self.add_template_variable(weight_name='template_gamma')
    if "center" not in kwargs or kwargs["center"] == True:
      self.add_template_variable(weight_name='template_beta')
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (self.input_spec, mixture_input_spec)

  def build(self, input_shape):
    BatchNormalization.build(self, input_shape)
    param_shape = (input_shape[self.axis[0]],)
    self.template_gamma = self.add_weight(
        name='template_gamma',
        shape=param_shape,
        dtype=self._param_dtype,
        initializer=self.gamma_initializer,
        regularizer=self.gamma_regularizer,
        constraint=self.gamma_constraint,
        trainable=True,
        experimental_autocast=False)
    self.template_beta = self.add_weight(
        name='template_beta',
        shape=param_shape,
        dtype=self._param_dtype,
        initializer=self.beta_initializer,
        regularizer=self.beta_regularizer,
        constraint=self.beta_constraint,
        trainable=True,
        experimental_autocast=False)

  def call(self, inputs, training=True):
    layer_inputs = inputs[0]
    mix_weights = inputs[1]
    # self.assign_all_mixture_values(mixture_weights=mix_weights)
    self.assign_mixture_value(name="template_beta", mixture_weights=mix_weights)
    self.assign_mixture_value(name="template_gamma", mixture_weights=mix_weights)
    norm_input = BatchNormalization.call(self, layer_inputs, training)

    input_shape = layer_inputs.shape
    num_channels = input_shape.dims[self.axis[0]].value
    reduction_axes = reduction_axes = [i for i in range(len(input_shape))
                                       if i not in self.axis]
    mean, var = nn.moments(norm_input, reduction_axes, keep_dims=True)
    scale = self._broadcast(self.template_gamma, input_shape)
    offset = self._broadcast(self.template_beta, input_shape)
    output = nn.batch_normalization(norm_input, mean, var, offset, scale,
                                    self.epsilon)
    self.reset_all_values()
    return output

  def _broadcast(self, tensor, input_shape):
    ndims = len(input_shape)
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    
    if tensor is not None and len(tensor.shape) != ndims:
        return tf.reshape(tensor, broadcast_shape)
    return tensor

  def get_config(self):
    config = super(WeighedBatchNormalizationSeparate, self).get_config()
    config['num_templates'] = self.num_templates
    return config

class WeightedConv2D(WeightedMixIn, Conv2D):
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
    kernel_initializer: An initializer for the convolution kernel templates.
    bias_initializer: An initializer for the bias vector templates. If None,
    the default initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel template
    bias_regularizer: Optional regularizer for the bias vector template.
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

  def __init__(self, num_templates=10, **kwargs):
    WeightedMixIn.__init__(self)
    Conv2D.__init__(self, **kwargs)
    self.num_templates = num_templates
    self.add_template_variable(weight_name='kernel')
    self.add_template_variable(weight_name='bias')
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (self.input_spec, mixture_input_spec)

  def call(self, inputs):
    layer_inputs = inputs[0]
    mix_weights = inputs[1]
    self.assign_mixture_value(name='kernel', mixture_weights=mix_weights)
    self.assign_mixture_value(name='bias', mixture_weights=mix_weights)
    output = super(WeightedConv2D, self).call(layer_inputs)
    self.reset_value('kernel')
    self.reset_value('bias')
    return output

  def get_config(self):
    config = super(WeightedConv2D, self).get_config()
    config['num_templates'] = self.num_templates
    return config

class WeightedDepthwiseConv2D(WeightedMixIn, DepthwiseConv2D):
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
    num_templates: integer number of templates. If the parameter `templates` is
      not None then `num_templates` specifies the number of templates to learn,
      otherwise `num_templates` parameter is ignored.
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

  def __init__(self, num_templates=10, **kwargs):
    DepthwiseConv2D.__init__(self, **kwargs)
    WeightedMixIn.__init__(self)
    self.num_templates = num_templates
    self.add_template_variable(weight_name='depthwise_kernel')
    self.add_template_variable(weight_name='bias')
    mixture_input_spec = InputSpec(ndim=1)
    self.input_spec = (self.input_spec, mixture_input_spec)


  def call(self, inputs):
    layer_inputs, mix_weights = inputs
    self.assign_mixture_value(name='depthwise_kernel',
                              mixture_weights=mix_weights)
    self.assign_mixture_value(name='bias', mixture_weights=mix_weights)
    output = super(WeightedDepthwiseConv2D, self).call(layer_inputs)
    self.reset_value('depthwise_kernel')
    self.reset_value('bias')
    return output

  def get_config(self):
    config = super(WeightedDepthwiseConv2D, self).get_config()
    config['num_templates'] = self.num_templates
    return config
