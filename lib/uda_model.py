"""Domain adaptation base model.
"""
import tensorflow as tf
import tensorflow.keras as tkf
import lib.multitask as mt

class BaseDomainAdaptationModel(tkf.Model):
  """Base unsupervised DA model.

  Arguments:
  classifier: classifier model.
  domains: domain names.
  target_domain: target domain name.
  num_layers: number of resblock layers.
  copy_mix_from: name of the source domain to copy mixture weights from. If
  None, the mixture weights will not be copied.
  """
  def __init__(self, classifier, domains, target_domain,
               num_layers, copy_mix_from=None, **args):
    super(BaseDomainAdaptationModel, self).__init__(**args)
    self.classifier = classifier
    self.domains = domains
    self.num_domains = len(self.domains)
    self.target_domain = target_domain
    if target_domain in domains:
      self.target_domain_idx = int(self.domains.index(self.target_domain))
    else:
      raise ValueError("The target domain is not in the domains list.")

    self.num_layers = num_layers
    self._input_shape = self.classifier.input_shape[0][1:]
    self._set_feature_extractor()
    if copy_mix_from is not None and copy_mix_from not in self.domains:
      raise ValueError("copy_mix_from domain is not in the list.")
    else:
      self.copy_mix_from = copy_mix_from

  def _set_mix_weights(self, model):
    """Sets mixture weights list as a model field.

    Arguments:
    model: model to which the mixture weights field will be added.
    """
    target_mix_weights = []
    prefix = "%s_mix" % self.target_domain
    for weight in model.trainable_weights:
      if prefix in weight.name:
        target_mix_weights.append(weight)
    model._target_mix_weights = target_mix_weights

  def _set_feature_extractor(self):
    """Sets late feature extractor of the model."""
    cls_inputs = [tkf.Input(name="%s_in_new" % x, shape=self._input_shape)
                  for x in self.domains]
    new_model_out = self.classifier(cls_inputs)
    try:
      block = (self.classifier.get_layer("feature_extractor")
                              .get_layer("weighted_resblock"))
      features_output_arr = []
      for idx in range(self.num_domains):
        out_idx = self.num_layers * (2 * self.num_domains - idx + 1) - 1
        outputs = block.get_output_at(out_idx)
        features_output_arr.append(outputs)
    except ValueError:
      # finding the last separate batch norm block
      feature_extr = self.classifier.get_layer("feature_extractor")
      last_bn_name = None
      bn_name = "weighted_res_block_separate_bn"
      for layer in feature_extr.layers:
        if bn_name in layer.name:
          last_bn_name = layer.name

      block = (self.classifier.get_layer("feature_extractor")
                              .get_layer(last_bn_name))
      features_output_arr = []
      for idx in range(self.num_domains):
        out_idx = 2 * self.num_domains - idx
        outputs = block.get_output_at(out_idx)
        features_output_arr.append(outputs)
    self.late_features = tf.keras.Model(cls_inputs, features_output_arr)
    # storing the target domain mixture weights
    self._set_mix_weights(self.late_features)

  def copy_mix_weights(self):
    """Copies mixture weights from source domain to target domain.
    """
    self.classifier = mt.copy_weights(
        self.classifier,
        source=self.copy_mix_from,
        target=self.target_domain,
        num_layers=self.num_layers,
        shared=True)

  def compile(self, cl_optimizer, cl_losses, cl_loss_weights):
    """Compiles the model.

    Arguments:
    cl_optimizer: classifier optimizer.
    cl_losses: classifier losses dictionary.
    cl_loss_weights: classifier loss weights dictionary.
    """
    super(BaseDomainAdaptationModel, self).compile()
    self.cl_optimizer = cl_optimizer
    self.cl_losses = cl_losses
    self.cl_loss_weights = cl_loss_weights
    loss_w = list(cl_loss_weights.values())
    self.source_domain_idx = list([i for i in range(self.num_domains)
                                   if loss_w[i] > 0])
    self.num_source = len(self.source_domain_idx)

  def call(self, data):
    """Calls classifier on the given data.
    """
    return self.classifier(data)

  def test_step(self, data):
    return self.classifier.test_step(data)

  def evaluate(self, *args, **kwargs):
    return self.classifier.evaluate(*args, **kwargs)

  def train_step(self, data):
    """Performs one training step. Should be overwritten by child class.
    """
    pass

