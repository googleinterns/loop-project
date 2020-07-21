import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def create_original_input_adapter(input_shape, depth):
  """Creates the original ResNet input adapter.
  
  Arguments:
  input_shape: input image shape.
  depth: output tensor depth."""
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(depth, kernel_size=3, strides=1, padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(1e-4))(inputs)
  x = layers.BatchNormalization()(x)
  outputs = layers.Activation('relu')(x)
  adapter = keras.Model(inputs, outputs, name='in_adapter')
  return adapter

def create_input_adapter_space2depth(input_shape, size=16, depth=40,
                                     activation=None):
  """Creates a space2depth input adapter module for the input image.
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

def create_input_adapter_strided(input_shape, filters=40, kernel=4,
                                 strides=4, activation=None):
  """Creates a strided input adapter module for the input image.
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


def create_output_adapter_depthwise(
    input_shape, block_size=None, pool_stride=None, activation='swish',
    depthwise=True, dropout=0):
  """Creates a depthwise output adapter module that processes tensors before
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

def create_output_adapter_v1(input_shape, dropout):
  """Returns an original ResnetV1 output adapter.
  
  Arguments:
  input_shape: input tensor shape.
  dropout: dropout (drop)."""
  inputs = keras.Input(input_shape)
  x = layers.AveragePooling2D(pool_size=8)(inputs)
  outputs = layers.Dropout(dropout)(x)
  model = tf.keras.Model(inputs, outputs, name='out_adapter')
  return model

def create_output_adapter_v2(input_shape, dropout):
  """Returns an original ResnetV2 output adapter.
  
  Arguments:
  input_shape: input tensor shape.
  dropout: dropout (drop)."""
  inputs = keras.Input(input_shape)
  x = layers.BatchNormalization()(inputs)
  x = layers.Activation('relu')(x)
  x = layers.AveragePooling2D(pool_size=8)(x)
  outputs = layers.Dropout(dropout)(x)
  model = tf.keras.Model(inputs, outputs, name='out_adapter')
  return model


def get_input_adapter(adapter_type, input_shape, tensor_size=16,
                     depth=40):
  """Returns an input adapter of given type.
  
  Arguments:
  adapter_type: input adapter type: `original`, `space2depth` or `strided`.
  input_shape: input image shape.
  tensor_size: output tensor size.
  depth: output tensor depth."""
  if adapter_type == 'original':
    return create_original_input_adapter(input_shape, depth)
  elif adapter_type == 'space2depth':
    return create_input_adapter_space2depth(
            input_shape, size=tensor_size, depth=depth, activation='relu')
  elif adapter_type == 'strided':
    k = max(input_shape[0] // tensor_size, input_shape[0] // tensor_size)
    return create_input_adapter_strided(input_shape, filters=depth,
                                              kernel=k, strides=k,
                                              activation='relu')
  raise ValueError("Given input adapter type is not supported.")

    
def get_output_adapter(adapter_type, input_shape, dropout=0,
                       out_filters=[128, 256]):
  """Returns an input adapter of given type.
  
  Arguments:
  adapter_type: output adapter type: `v1`, `v2`, `isometric` or `dethwise`.
  input_shape: input tensor shape.
  dropout: dropout (drop).
  out_filters: list of two integers representing number of conv filters
    in the isometric adapter."""
  if adapter_type == 'v1':
    return create_output_adapter_v1(input_shape, dropout)
  elif adapter_type == 'v2':
    return create_output_adapter_v2(input_shape, dropout)
  elif adapter_type == 'isometric':
    return create_output_adapter_isometric(
            input_shape, filters=out_filters, pool_size=8, activation='relu',
            dropout=dropout)
  elif adapter_type == 'dethwise':
    return create_output_adapter_depthwise(
            input_shape, block_size=2, pool_stride=None,
            activation='relu', depthwise=True, dropout=dropout)
  else:
        raise ValueError("Given output adapter type is not supported.")
    