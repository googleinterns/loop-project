"""Domain adaptation via moment matching.
"""
import tensorflow as tf
import tensorflow.keras as tkf

class MomentMatchingClassifier(tkf.Model):
  """Moments matching classifier for unsupervised domain adaptation.

  Arguments:
  classifier: classifier model.
  target_domain: target domain name.
  num_layers: number of resblock layers in the classifier.
  """
  def __init__(self, classifier, domains, target_domain,
               num_layers, **args):
    super(MomentMatchingClassifier, self).__init__(**args)
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
      #finding the last separate batch norm block
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
    target_mix_weights = []
    prefix = "%s_mix" % self.target_domain
    for weight in self.late_features.trainable_weights:
      if prefix in weight.name:
        target_mix_weights.append(weight)
    self.late_features._target_mix_weights = target_mix_weights


  def compile(self, cl_optimizer, mm_optimizer, cl_losses, cl_loss_weights,
              mm_loss_weight=1):
    """Compiles the model.

    Arguments:
    cl_optimizer: classifier optimizer.
    mm_optimizer: moment matching optimizer.
    cl_losses: classifier losses dictionary.
    cl_loss_weights: classifier loss weights dictionary.
    mm_loss_weight: moment matching loss weights.
    """
    super(MomentMatchingClassifier, self).compile()
    self.classifier.compile(optimizer=cl_optimizer,
                            loss=cl_losses,
                            loss_weights=cl_loss_weights,
                            metrics="acc")
    self.mm_optimizer = mm_optimizer
    self.mm_loss_weight = mm_loss_weight
    loss_w = list(cl_loss_weights.values())
    self.source_domain_idx = list([i for i in range(self.num_domains)
                                   if loss_w[i] > 0])
    self.num_source = len(self.source_domain_idx)

  def call(self, data):
    """Calls classifier on the given data.
    """
    return self.classifier(data)

  def train_step(self, data):
    images, _ = data
    cl_results = self.classifier.train_step(data)
    # Train the mixture weights to match the moments
    with tf.GradientTape() as tape:
      features = self.late_features(images)
      target_features = features[self.target_domain_idx]
      source_features = [features[x] for x in self.source_domain_idx]

      # compute the moments
      mean_target = tf.reduce_mean(target_features, axis=0)
      mean_sq_target = tf.reduce_mean(tf.math.square(target_features),
                                      axis=0)
      mean_source = [tf.reduce_mean(x, axis=0) for x in source_features]
      mean_sq_source = [tf.reduce_mean(tf.math.square(x), axis=0)
                        for x in source_features]
      mean_losses = [tf.nn.l2_loss(mean_target - x) for x in mean_source]
      mean_loss = tf.add_n(mean_losses) / self.num_source

      mean_sq_losses = [tf.nn.l2_loss(mean_sq_target - x)
                        for x in mean_sq_source]
      mean_sq_loss = tf.add_n(mean_sq_losses) / self.num_source
      cross_source_loss = 0.
      cross_source_sq_loss = 0.

      if self.num_source > 1:
        for i in range(self.num_source - 1):
          for j in range(i + 1, self.num_source, 1):
            cross_source_loss += tf.nn.l2_loss(
                mean_source[i] - mean_source[j])
            cross_source_sq_loss += tf.nn.l2_loss(
                mean_sq_source[i] - mean_sq_source[j])

      n_bin = 2. / self.num_source * (self.num_source - 1)
      cross_source_loss *= n_bin
      cross_source_sq_loss *= n_bin

      total_mm_loss = self.mm_loss_weight * (mean_loss + mean_sq_loss +
                                             cross_source_loss +
                                             cross_source_sq_loss)

    grads = tape.gradient(total_mm_loss, self.late_features._target_mix_weights)
    self.mm_optimizer.apply_gradients(
        zip(grads, self.late_features._target_mix_weights))

    result = cl_results
    result["mean_loss"] = mean_loss
    result["cross_source_loss"] = cross_source_loss

    return result

  def test_step(self, data):
    return self.classifier.test_step(data)

def evaluate(self, *args, **kwargs):
  return self.classifier.evaluate(*args, **kwargs)
