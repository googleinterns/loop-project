import tensorflow as tf
from tensorflow.keras import layers
import lib.io_adapters as ioad
from lib import weighted_resblock as wb
from lib.resnet_parameters import ResNetParameters

# def shared_resnet(input_shape, num_layers=16, num_classes=10, num_templates=4,
#                   depth=40, in_adapter="strided", out_adapter="isometric",
#                   tensor_size=8, kernel_regularizer=1e-5, dropout=0,
#                   out_filters=None, with_head=True,
#                   mixture_weights_as_input=False, separate_bn=False,
#                   name="model", activation="relu"):
def shared_resnet(parameters: ResNetParameters) -> tf.keras.Model:
  """ResNet v1 with custom resblock model builder

  Arguments
    parameters: ResNetParameters object.
  Returns
      Keras model instance"""
  layer_input = tf.keras.Input(shape=parameters.input_shape,
                               name="layer_input")
  in_adapter = ioad.get_input_adapter(
      parameters.in_adapter, parameters.input_shape,
      parameters.tensor_size, parameters.depth,
      activation=parameters.activation)
  x = in_adapter(layer_input)

  if parameters.mixture_weights_as_input:
    mix_input = [tf.keras.Input(shape=(parameters.num_templates,),
                                name="mix_%d" % i,
                                batch_size=1)
                 for i in range(parameters.num_layers)]
    inputs = [layer_input, *mix_input]
    mix_weights = mix_input
  else:
    inputs = layer_input
    # creating new mixture weights
    xi_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    mix_weights = []
    for _ in range(parameters.num_layers):
      xi = wb.MixtureWeight(num_templates=parameters.num_templates,
                            initializer=xi_initializer)(x)
      mix_weights.append(xi)

  # Instantiate the stack of residual units
  w_res_block = wb.WeightedResBlock(kernel_size=3, expansion_factor=6,
                                    activation=parameters.activation,
                                    num_templates=parameters.num_templates,
                                    kernel_reg=parameters.kernel_regularizer,
                                    name="weighted_resblock")
  # Instantiate the stack of residual units
  for i in range(parameters.num_layers):
    xi = mix_weights[i]
    xi = layers.Lambda(
        lambda x: tf.reshape(x, [parameters.num_templates,]))(xi)
    if parameters.separate_bn:
      block = wb.WeightedResBlockSeparateBN(w_res_block)
      x = block([x, xi])
    else:
      x = w_res_block([x, xi])
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
