import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import lib.io_adapters as ioad
from lib import weighted_resblock as wb

def shared_resnet(input_shape, num_layers=16, num_classes=10, num_templates=4,
                  depth=40, in_adapter="original", out_adapter="v1",
                  tensor_size=8, kernel_regularizer=1e-5, dropout=0,
                  out_filters=None, with_head=True,
                  mixture_weights_as_input=False, name="model"):
  """ResNet v1 with custom resblock model builder

  Arguments
      input_shape (tensor): shape of input image tensor
      num_layers (int): number of residual blocks
      num_classes (int): number of classes
      num_templates (int): number of templates
      depth: depth of resblock tensors
      in_adapter (string): input adapter architecture. The options are
        `original`, `space2depth` and `strided`.
      out_adapter (string): output adapter architecture. The options are
        `v1`, `v2`, `isometric` and `dethwise`.
      kernel_regularizer: kernel regularization parameter.
      dropout: dropout parameter (drop).
      out_filters: a list of two integers that represent number of filters
          in conv layers of isometric adapter.
      with_head: if True, the model will have a head (top dense layer).
      mixture_weights_as_input: if False, the mixture weights will be created
          with the model, otherwise they are treated as model input.
      name: model name.
  Returns
      model (Model): Keras model instance"""
  if mixture_weights_as_input:
    layer_input = tf.keras.Input(shape=input_shape, name='layer_input')
    mix_input = [tf.keras.Input(shape=(num_templates,), name='mix_%d' % i,
                                batch_size=1)
                                for i in range(num_layers)]
    inputs = [layer_input, *mix_input]
  else:
    layer_input = tf.keras.Input(shape=input_shape, name='input')
    inputs = layer_input
    
  in_adapter = ioad.get_input_adapter(
      in_adapter, input_shape, tensor_size, depth)
  x = in_adapter(layer_input)

  # Instantiate the stack of residual units
  w_res_block = wb.WeightedResBlock(kernel_size=3, expansion_factor=6,
                                    activation='swish',
                                    num_templates=num_templates,
                                    kernel_reg=kernel_regularizer,
                                    name="weighted_resblock")
  if mixture_weights_as_input:
      mix_weights = mix_input
  else:
    xi_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    mix_weights = []
    for _ in range(num_layers):
      xi = wb.MixtureWeight(num_templates=num_templates,
                            initializer=xi_initializer)(x)
      mix_weights.append(xi)

  # Instantiate the stack of residual units
  for i in range(num_layers):
    xi = mix_weights[i]
    xi = layers.Lambda(lambda x: tf.reshape(x, [num_templates,]))(xi)
    x = w_res_block([x, xi])
    
  t_shape = input_shape[0] if in_adapter == 'original' else tensor_size
  in_shape = (t_shape, t_shape, depth)
  out_adapter = ioad.get_output_adapter(out_adapter, in_shape, dropout,
                                        out_filters)
  x = out_adapter(x)
  x = layers.Flatten()(x)
  if with_head:
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)
  else:
    outputs = x

  # Instantiate model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
  return model
