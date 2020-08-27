"""Discriminative domain adaptation module.
"""
import tensorflow as tf
import tensorflow.keras as tkf

def get_discriminator(input_shape, num_layers=3, activation="relu", dropout=0):
  """Returns the disctiminator model.

  Arguments:
  input_shape: shape of the input tensor.
  num_layers: number of convolutional layers.
  activaiton: activation function (default is relu).
  """
  discriminator = tkf.Sequential(name="discriminator")
  discriminator.add(tkf.layers.InputLayer(input_shape))
  for x in range(num_layers):
    discriminator.add(
        tkf.layers.Conv2D(16 * (x + 1), (3, 3), strides=(1, 1),
                          padding="same", activation=activation))

  discriminator.add(tkf.layers.AveragePooling2D(pool_size=(8, 8)))
  discriminator.add(tkf.layers.Flatten())
  discriminator.add(tkf.layers.Dropout(dropout))
  discriminator.add(tkf.layers.Dense(1, activation="sigmoid"))
  return discriminator

class AdversarialClassifier(tkf.Model):
  """Discriminative classifier for unsupervised domain adaptation.

  Consists of a classifier network that is trained to perform classification on
  the source domain, and domain disriminator that, given late features from the
  classifier, predicts whether the features come from source or target domain.
  The target domain mixture weights are optimized to fool the discriminator.

  Arguments:
  classifier: classifier model.
  discriminator: discriminator model.
  domains: list of domain names.
  target_domain: target domain name.
  num_layers: number of resblock layers in the classifier.
  """
  def __init__(self, classifier, discriminator, domains, target_domain,
               num_layers, **args):
    super(AdversarialClassifier, self).__init__(**args)
    self.classifier = classifier
    self.discriminator = discriminator
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
      for layer in feature_extr.layers:
        if "weighted_res_block_separate_bn" in layer.name:
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


  def compile(self, d_optimizer, cl_optimizer, cl_losses, cl_loss_weights,
              adv_optimizer, loss_fn, d_loss_weight=1, adv_loss_weight=1):
    """Compiles the model.

    Arguments:
    d_optimizer: discriminator optimizer.
    cl_optimizer: classifier optimizer.
    adv_optimizer: adversarial optimizer.
    cl_losses: classifier losses dictionary.
    cl_loss_weights: classifier loss weights dictionary.
    loss_fn: discriminator loss function.
    d_loss_weight: discriminator loss weights.
    adv_loss_weight: adversarial loss weights.
    """
    super(AdversarialClassifier, self).compile()
    self.classifier.compile(optimizer=cl_optimizer,
                            loss=cl_losses,
                            loss_weights=cl_loss_weights,
                            metrics="acc")
    self.d_optimizer = d_optimizer
    self.adv_optimizer = adv_optimizer
    self.loss_fn = loss_fn
    self.d_loss_weight = d_loss_weight
    self.adv_loss_weight = adv_loss_weight
    loss_w = list(cl_loss_weights.values())
    self.source_domain_idx = list([i for i in range(self.num_domains)
                                   if loss_w[i] > 0])

  def call(self, data):
    """Calls classifier on the given data.
    """
    return self.classifier(data)

  def train_step(self, data):
    images, _ = data
    batch_size = tf.shape(images[0])[0]

    cl_results = self.classifier.train_step(data)
    # Compute late features
    features = self.late_features(images)
    target_features = features[self.target_domain_idx]
    source_features = [features[x] for x in self.source_domain_idx]
    combined_source_features = tf.concat(source_features, axis=0)

    label_shape = (batch_size, 1)
    # discriminator labels
    d_labels_source = tf.zeros((len(self.source_domain_idx) * batch_size, 1))
    d_labels_target = tf.ones(label_shape)
    d_labels_source += 0.1 * tf.random.uniform(tf.shape(d_labels_source))
    d_labels_target += 0.1 * tf.random.uniform(tf.shape(d_labels_target))

    # Train the discriminator to discriminate source and target features
    with tf.GradientTape() as tape:
      pred_source = self.discriminator(combined_source_features)
      pred_target = self.discriminator(target_features)
      d_loss_source = self.loss_fn(d_labels_source, pred_source)
      d_loss_target = self.loss_fn(d_labels_target, pred_target)
      d_loss_weighted = d_loss_source + d_loss_target
      d_loss_weighted *= self.d_loss_weight * 0.5
    grads = tape.gradient(d_loss_weighted, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(
        zip(grads, self.discriminator.trainable_weights))

    # Assemble labels that say "all source images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the mixture weights to fool the discriminatoer
    with tf.GradientTape() as tape:
      features = self.late_features(images)
      predictions = self.discriminator(features[self.target_domain_idx])
      adv_loss = self.loss_fn(misleading_labels, predictions)
      adv_loss_weighted = self.adv_loss_weight * self.loss_fn(misleading_labels,
                                                              predictions)
    grads = tape.gradient(adv_loss_weighted,
                          self.late_features._target_mix_weights)
    self.adv_optimizer.apply_gradients(
        zip(grads, self.late_features._target_mix_weights))

    result = cl_results
    result["d_loss_source"] = d_loss_source
    result["d_loss_target"] = d_loss_target
    result["adv_loss"] = adv_loss

    return result

  def test_step(self, data):
    return self.classifier.test_step(data)

  def evaluate(self, *args, **kwargs):
    return self.classifier.evaluate(*args, **kwargs)
