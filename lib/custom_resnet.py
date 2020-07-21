import tensorflow as tf
import lib.io_adapters as ioad
import lib.res_block as block
from tensorflow.keras import layers


def resnet(input_shape, num_layers=20, num_classes=10, depth=40,
           in_adapter="original", out_adapter="v1", tensor_size=8, 
           kernel_regularizer=1e-5, dropout=0, out_filters=[128, 256],
           with_head=True, name="model"):
  """ResNet v1 with custom resblock model builder 

  Arguments
    input_shape (tensor): shape of input image tensor
    num_layers (int): number of residual blocks
    num_classes (int): number of classes
    depth: depth of resblock tensors
    in_adapter (string): input adapter architecture. The options are
      `original`, `space2depth` and `strided`.
    out_adapter (string): output adapter architecture. The options are
      `v1`, `v2`, `isometric` and `dethwise`.
    kernel_regularizer: kernel regularization parameter.
    dropout: dropout parameter (drop).
    out_filters: list of two integers representing number of conv filters in
      the isometric adapter.
    with_head: whether to add a top dense layer.
    name: model name.
  Returns
    model (Model): Keras model instance
    """

  inputs = tf.keras.Input(shape=input_shape, name='input')
  in_adapter = ioad.get_input_adapter(
      in_adapter, input_shape, tensor_size, depth)
  x = in_adapter(inputs)

  # Instantiate the stack of residual units
  for stack in range(num_layers):
    x = block.ResBlock(kernel_size=3, expansion_factor=6,
                       activation='swish',
                       kernel_reg=kernel_regularizer)(x)

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