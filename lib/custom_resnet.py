import tensorflow as tf
import lib.io_adapters as ioad
import lib.res_block as block
from tensorflow.keras import layers
from lib.resnet_parameters import ResNetParameters

# def resnet(input_shape, num_layers=20, num_classes=10, depth=40,
#            in_adapter="original", out_adapter="v1", tensor_size=8,
#            kernel_regularizer=1e-5, dropout=0, out_filters=[128, 256],
#            with_head=True, name="model", activation="relu"):
def resnet(parameters: ResNetParameters) -> tf.keras.Model:
  """ResNet v1 with custom resblock model builder

  Arguments
    parameters: ResNetParameters instance.
  Returns
    model (Model): Keras model instance
  """

  inputs = tf.keras.Input(shape=parameters.input_shape, name="input")
  in_adapter = ioad.get_input_adapter(
      parameters.in_adapter, parameters.input_shape, parameters.tensor_size,
      parameters.depth, activation=parameters.activation)
  x = in_adapter(inputs)

  # Instantiate the stack of residual units
  for _ in range(parameters.num_layers):
    x = block.ResBlock(kernel_size=3, expansion_factor=6,
                       activation=parameters.activation,
                       kernel_reg=parameters.kernel_regularizer)(x)

  if "original" in parameters.in_adapter:
    t_shape = parameters.input_shape[0]
  else:
    t_shape = parameters.tensor_size
  in_shape = (t_shape, t_shape, parameters.depth)
  out_adapter = ioad.get_output_adapter(
      parameters.out_adapter, in_shape, parameters.dropout,
      parameters.out_filters, activation=parameters.activation)
  x = out_adapter(x)
  x = layers.Flatten()(x)
  if parameters.with_head:
    outputs = layers.Dense(
        parameters.num_classes, activation="softmax",
        kernel_initializer="he_normal")(x)
  else:
    outputs = x

  # Instantiate model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=parameters.name)
  return model
