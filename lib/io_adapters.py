import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_input_adapter(input_shape, size=16, depth=40, activation=None):
  """Creates an input adapter module for the input image.
  The input adapter transforms input image of given shape
  into a tensor of target shape.

  Arguments: 
    input_shape: shape of input image (HxWxC). Image width and height
    must be devidible by size. H and W must be greater than or equal to `size`.
    size: height and width of the output tensor after space2depth operation.
    depth: number of channels in the output tensor.
    activation: conv layer activation function."""
  h, w, _ = input_shape
  if h < size or w < size:
    raise ValueError('Input height and width should be greater than `size`.')
  # `block_size` of space2depth
  block_size = min(h / size, w / size)

  # creating an adapter model
  inputs = keras.Input(shape=input_shape)
  s2d = tf.nn.space_to_depth(inputs, block_size)
  outputs = layers.Conv2D(filters=depth,
                          kernel_size=1, activation=activation)(s2d)
  model = keras.Model(inputs, outputs, name='in_adapter')
  return model


def create_output_adapter(input_shape, block_size=None, pool_stride=None,
                          activation='swish', depthwise=True, dropout=0):
  """Creates an output adapter module that processes tensors before
  passing them to fully connected layers.
  Arguments:
    input_shape: shape of the input tensor (HxWxC).
    block_size: tensor height and width after average pooling. Default
    value is 4.
    pool_stride: stride of average pooling.
    activation: activation function.
    depthwise: whether to use depthwise convolution.
    dropout: dropout parameter (drop)."""
  if not block_size:
    block_size = 4

  if not isinstance(block_size, int) or block_size < 1:
    raise ValueError("block_size must be a positive integer.")

  if pool_stride is not None and (not isinstance(pool_stride, int) or
                                  pool_stride < 1):
    raise ValueError("pool_stride be a positive integer or None.")

  if len(input_shape) != 3:
    raise ValueError("input_shape must be a tuple of size 3.")

  h, w, _ = input_shape
  inputs = tf.keras.Input(shape=input_shape)
  kernel_size = (tf.round(h / block_size), tf.round(w / block_size))

  x = layers.AveragePooling2D(pool_size=kernel_size,
                              strides=pool_stride,
                              padding='valid')(inputs)
  if depthwise:
    x = layers.DepthwiseConv2D(kernel_size=1,
                               activation=activation)(x)
  else:
    x = layers.Activation(activation)(x)
  x = layers.Dropout(dropout)(x)
  x = layers.Flatten(data_format='channels_last')(x)
  outputs = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
  model = keras.Model(inputs, outputs, name='out_adapter')
  return model

def create_output_adapter_isometric(input_shape, filters=None, pool_size=None,
                        activation='relu', dropout=0):
  """ Creates an output adapter module that processes tensors before
  passing them to fully connected layers.

  Arguments:
  input_shape: shape of the input tensor (HxWxC).
  filters: a list or tuple of two integers representing number of filters in
    1x1 convolutions.
  pool_size: pool size of average pooling layer.
  activation: activation function.
  dropout: dropout parameter (drop)."""

  if filters is None: 
    filters = [960, 1280]

  if pool_size is None:
    pool_size = 8
  
  if len(filters) != 2:
    raise ValueError("filters must be a list or tuple of two.")

  if (not isinstance(pool_size, int) or pool_size < 1):
    raise ValueError("pool_size be a positive integer or None.")
  
  if len(input_shape) != 3:
    raise ValueError("input_shape must be a tuple of size 3.")

  h, w, _ = input_shape
  if h == w:
    pool_size = [h, w]
  else: 
    pool_size = [pool_size, pool_size]

  inputs = tf.keras.Input(shape=input_shape)
  x = layers.Conv2D(kernel_size=1, strides=1, filters=filters[0],
                    activation=activation)(inputs)

  x = layers.AveragePooling2D(pool_size=pool_size, strides=1)(x)

  x = layers.Conv2D(kernel_size=1, strides=1, filters=filters[1],
                    padding='valid', activation=activation)(x)
  outputs = layers.Dropout(dropout)(x)
  model = tf.keras.Model(inputs, outputs, name='out_adapter')
  return model

def create_input_adapter_strided(input_shape, filters=40, kernel=4,
                                 strides=4, activation=None):
  """Creates an input adapter module for the input image.
  The input adapter transforms input image of given shape
  into a tensor of target shape.

  Arguments:
    input_shape: shape of input tensor.
    filters: number of filters of convolution.
    kernel: convolution kernel size.
    strides: Stride of convolution.
    activation: conv layer activation function."""

  inputs = tf.keras.Input(shape=input_shape)
  x = layers.Conv2D(filters=filters,
                    kernel_size=kernel, strides=strides,
                    activation=None)(inputs)
  x = layers.BatchNormalization()(x)
  outputs = layers.Activation(activation)(x)
  model = tf.keras.Model(inputs, outputs, name='in_adapter')
  return model
