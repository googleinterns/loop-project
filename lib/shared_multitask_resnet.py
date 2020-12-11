import tensorflow as tf
from tensorflow.keras import layers
import lib.io_adapters as ioad
from lib import weighted_resblock as wb
from lib.resnet_parameters import ResNetParameters

def shared_multitask_resnet(domain_list, 
                            parameters: ResNetParameters) -> tf.keras.Model:
  """Shared ResNet that uses multitask residual block.

  Arguments
    domain_list: list of domain names.
    parameters: ResNetParameters object.
  Returns
      Keras model instance"""
  num_domains = len(domain_list)
  layer_input = [tf.keras.Input(
      shape=parameters.input_shape,
      name="%s_layer_input" % dname) for dname in domain_list]
  in_adapter = ioad.get_input_adapter(
      parameters.in_adapter, parameters.input_shape,
      parameters.tensor_size, parameters.depth,
      activation=parameters.activation)
  mix_input = []
  for dname in domain_list:
     for i in range(parameters.num_layers):
      mix_input.append(tf.keras.Input(shape=(parameters.num_templates,),
                              name="%s_mix_%i_input" % (dname, i),
                              batch_size=1))
  inputs = [layer_input, mix_input]
  mix_weights = mix_input
  
  # Instantiate the stack of residual units
  w_res_block = wb.WeightedResBlock(
      kernel_size=3, expansion_factor=6, activation=parameters.activation,
      num_templates=parameters.num_templates,
      kernel_reg=parameters.kernel_regularizer,
      name="weighted_resblock")
  
  x = [in_adapter(layer_input[idx]) for idx in range(num_domains)]
  # Instantiate the stack of residual units
  for i in range(parameters.num_layers):
    xi = [mix_weights[i + idx * parameters.num_layers]
          for idx in range(num_domains)]
    xi = [layers.Lambda(
        lambda a: tf.reshape(a, [parameters.num_templates,]))(xi[idx]) 
        for idx in range(num_domains)]
    block = wb.WeightedMultitaskResBlock(w_res_block, domain_list=domain_list)
    x = block([x, xi])
    
  if "original" in parameters.in_adapter:
    t_shape = parameters.input_shape[0]
  else:
    t_shape = parameters.tensor_size
  in_shape = (t_shape, t_shape, parameters.depth)
  out_adapter = ioad.get_output_adapter(
      parameters.out_adapter, in_shape, parameters.dropout,
      parameters.out_filters, activation=parameters.activation)
  
  x = [out_adapter(x[idx]) for idx in range(num_domains)]
  x = [layers.Flatten()(x[idx]) for idx in range(num_domains)]
  if parameters.with_head:
    head = layers.Dense(
        parameters.num_classes, activation="softmax",
        kernel_initializer="he_normal")
    outputs = [head(x[idx]) for idx in range(num_domains)]
  else:
    outputs = x

  # Instantiate model.
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=parameters.name)
  return model