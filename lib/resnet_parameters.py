from dataclasses import dataclass, field
import tensorflow as tf

@dataclass
class ResNetParameters:
  """Stores ResNet parameters.

  Arguments:
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
  separate_bn: if True, separate batch normalization layers will be used
      for each layer.
  name: model name.
  """
  input_shape: tuple = (32, 32, 3)
  num_layers: int = 16
  num_classes: int = 10
  num_templates: int = 4
  depth: int = 40
  tensor_size: int = 16
  in_adapter: str = "strided"
  out_adapter: str = "isometric"
  out_filters: list = field(default_factory=list)
  dropout: float = 0.1
  kernel_regularizer: float = 1e-5
  with_head: bool = True
  name: str = "model"
  activation: callable = tf.nn.relu
  mixture_weights_as_input: bool = False
  separate_bn: bool = False

  def init_from_args(self, args):
    self.input_shape = (args.reshape_to, args.reshape_to, 3)
    self.num_layers = args.num_blocks
    self.tensor_size = args.size
    self.depth = args.depth
    self.separate_bn = args.sep_bn > 0
    self.num_templates = args.num_templates
    self.dropout = args.dropout
    self.in_adapter = args.in_adapter_type
    self.out_adapter = args.out_adapter_type
    self.out_filters = [args.out_filter_base, 2 * args.out_filter_base]
    self.kernel_regularizer = args.kernel_reg

