import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class ResBlock(tf.keras.Model):
  """A ResBlock module class with expansion, depthwise conv and projection.

  In this ResBlock, standard 2D convolutions are replaced by 1x1 convolution
  that expands the input tensor along the channel dimension, depthwise
  convolution and 1x1 convolution that projects the tensor back to the original
  number of channels.

   Args:
     kernel_size: size of the depthwise convolution kernel
     expansion_factor: expansion factor of the first 1x1 convolution.
         e.g., if the input tensor has N channels, then the first 1x1
         convolution layer will expand it to expansion_factor*N channels.
     activation: activation function. Supported functions: 'relu',
         'relu6', 'lrelu', 'swish'.
     kernel_reg: kernel regularizer parameter.
  """
  def __init__(self, kernel_size=3, expansion_factor=6, activation="relu",
               kernel_reg=1e-5):
    super(ResBlock, self).__init__(name="")
    if expansion_factor < 1:
      raise ValueError("The expansion factor value should be "
                       "greater than or equal to one.")

    self.expansion_factor = expansion_factor
    self.activation = self.set_activation_fn(activation)
    self.kernel_size = kernel_size
    self.kernel_reg = kernel_reg

  def build(self, input_shape):
    input_channel = input_shape[-1]
    self.expanded_channel = input_channel*self.expansion_factor
    self.conv1 = layers.Conv2D(
        self.expanded_channel, kernel_size=1, strides=(1, 1), padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(self.kernel_reg))
    self.bn1 = layers.BatchNormalization()
    self.conv2 = layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                        strides=(1, 1), padding="same")
    self.bn2 = layers.BatchNormalization()
    self.conv3 = layers.Conv2D(
        input_channel, kernel_size=1, strides=(1, 1), padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(self.kernel_reg))
    self.bn3 = layers.BatchNormalization()

  def call(self, input_tensor, training=True):
    x = self.conv1(input_tensor)
    x = self.bn1(x, training=training)
    x = self.activation(x)

    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.activation(x)

    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x += input_tensor
    return x

  def set_activation_fn(self, activation):
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
