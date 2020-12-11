"""Domain adaptation via moment matching.
"""
import tensorflow as tf
import tensorflow.keras as tkf
from lib.uda_model import BaseDomainAdaptationModel
from lib import multitask as mt

class MomentMatchingClassifier(BaseDomainAdaptationModel):
  """Moment matching classifier for unsupervised domain adaptation.

  Consists of a classifier network that is trained to perform classification on
  the source domain, and domain disriminator that, given late features from the
  classifier, predicts whether the features come from source or target domain.
  The target domain mixture weights are optimized to fool the discriminator.

  Arguments:
  classifier: classifier model.
  target_domain: target domain name.
  num_layers: number of resblock layers in the classifier.
  """
  def __init__(self, *args, **kwargs):
    super(MomentMatchingClassifier, self).__init__(*args, **kwargs)
    self.copy_mix_weights()

  def compile(self, mm_optimizer, mm_loss_weight, *args, **kwargs):
    """Compiles the model.

    Arguments:
    cl_optimizer: classifier optimizer.
    mm_optimizer: moment matching optimizer.
    cl_losses: classifier losses dictionary.
    cl_loss_weights: classifier loss weights dictionary.
    mm_loss_weight: moment matching loss weights.
    """
    super(MomentMatchingClassifier, self).compile(*args, **kwargs)
    if self.classifier is not None:
      self.classifier.compile(optimizer=self.cl_optimizer,
                              loss=self.cl_losses,
                              loss_weights=self.cl_loss_weights,
                              metrics="acc")
    self.mm_optimizer = mm_optimizer
    self.mm_loss_weight = mm_loss_weight

  def call(self, data):
    """Calls classifier on the given data.
    """
    return self.classifier(data)


  def match_moments(self, images):
    # Train the mixture weights to match the moments
    with tf.GradientTape() as tape:
      features = self.late_features(images)
      target_features = features[self.target_domain_idx]
      source_features = [features[x] for x in self.source_domain_idx]

      # compute the moments
      mean_target = tf.reduce_mean(target_features, axis=0)
      mean_sq_target = tf.reduce_mean(tf.math.square(target_features), axis=0)
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
      # compute cross-source discances
      if self.num_source > 1:
        for i in range(self.num_source - 1):
          for j in range(i + 1, self.num_source, 1):
            cross_source_loss += tf.nn.l2_loss(
                mean_source[i] - mean_source[j])
            cross_source_sq_loss += tf.nn.l2_loss(
                mean_sq_source[i] - mean_sq_source[j])

      binomial_c = 2. / (self.num_source * (self.num_source - 1))
      cross_source_loss *= binomial_c
      cross_source_sq_loss *= binomial_c

      total_mm_loss = 0.0005 * (mean_loss + mean_sq_loss +
                                cross_source_loss +
                                cross_source_sq_loss)

    grads = tape.gradient(total_mm_loss,
                          self.da_trainable_parameters)
    self.mm_optimizer.apply_gradients(
        zip(grads, self.da_trainable_parameters))
    return {"mean_loss": 0.0005 * mean_loss,
            "cross_source_loss": 0.0005 * cross_source_loss}

  def train_step(self, data):
    images, _ = data
    cl_results = {}
    if self.train_classifier:
      cl_results = self.classifier.train_step(data)
    mm_results = self.match_moments(images)

    return cl_results.update(mm_results)


class MomentMatchingClassifierV2(MomentMatchingClassifier):
  """Discriminative classifier for unsupervised domain adaptation.

  Consists of a classifier network that is trained to perform classification on
  the source domain, and domain disriminator that, given late features from the
  classifier, predicts whether the features come from source or target domain.
  The target domain mixture weights are optimized to fool the discriminator.

  Arguments:
  classifier: classifier model.
  target_domain: target domain name.
  num_layers: number of resblock layers in the classifier.
  num_classes: number of classes.
  """
  def __init__(self, num_classes, *args, **kwargs):
    super(MomentMatchingClassifierV2, self).__init__(*args, **kwargs)
    self.num_classes = num_classes
    self.headless = self.classifier
    self.classifier = None
    self._set_heads()
    self._set_classifiers()
    self._set_mix_weights(self.headless)

  def _rename(self, inputs, name):
    return tkf.layers.Lambda(lambda x: x, name=name)(inputs)

  def _set_classifiers(self):
    """Sets classifiers sharing the same feature extractor."""
    cl1_inputs = [tkf.Input(name="%s_in_cl1" % x, shape=self._input_shape)
                  for x in self.domains]
    features = self.headless(cl1_inputs)
    outputs_1 = []
    for i in range(self.num_domains):
      output = self._rename(
          self.head_1(features[i]), "%s_out" % self.domains[i])
      outputs_1.append(output)
    self.classifier_1 = tf.keras.Model(cl1_inputs, outputs_1)

    cl2_inputs = [tkf.Input(name="%s_in_cl2" % x, shape=self._input_shape)
                  for x in self.domains]
    features = self.headless(cl2_inputs)
    outputs_2 = []
    for i in range(self.num_domains):
      output = self._rename(
          self.head_2(features[i]), "%s_out" % self.domains[i])
      outputs_2.append(output)
    self.classifier_2 = tf.keras.Model(cl2_inputs, outputs_2)

  def _set_heads(self):
    """Creates two logits layers.
    """
    self.head_1 = tf.keras.layers.Dense(
        self.num_classes, activation="softmax",
        kernel_initializer="he_normal", name="head_1")
    self.head_2 = tf.keras.layers.Dense(
        self.num_classes, activation="softmax",
        kernel_initializer="he_normal", name="head_2")

  def copy_mix_weights(self):
    """Copies mixture weights from source domain to target domain.
    """
    if not self.copy_mix_from is None:
      model = self.headless if self.classifier is None else self.classifier
      _ = mt.copy_weights(
          model,
          source=self.copy_mix_from,
          target=self.target_domain,
          num_layers=self.num_layers,
          shared=True)


  def compile(self, cl1_optimizer, cl2_optimizer, cl_losses,
              cl_loss_weights, *args, **kwargs):
    """Compiles the model.

    Arguments:
    cl_optimizer: classifier optimizer.
    mm_optimizer: moment matching optimizer.
    cl_losses: classifier losses dictionary.
    cl_loss_weights: classifier loss weights dictionary.
    mm_loss_weight: moment matching loss weights.
    """
    super(MomentMatchingClassifierV2, self).compile(
        cl_optimizer=cl1_optimizer,
        cl_losses=cl_losses,
        cl_loss_weights=cl_loss_weights,
        *args, **kwargs)
    self.classifier_1.compile(optimizer=cl1_optimizer,
                              loss=cl_losses,
                              loss_weights=cl_loss_weights,
                              metrics="acc")
    self.classifier_2.compile(optimizer=cl2_optimizer,
                              loss=cl_losses,
                              loss_weights=cl_loss_weights,
                              metrics="acc")

    self.discrepancy_loss = tf.keras.losses.MeanAbsoluteError()
    self.min_discr_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4)
    self.max_discr_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4)


  def call(self, data):
    """Calls classifier on the given data.
    """
    pred = 0.5 * (self.classifier_1(data) + self.classifier_2(data))
    return pred

  def classifier_train_step(self, data):
    """Train step of the classifier and both heads.

    Arguments:
    data: data batch.
    """
    cl1_results = self.classifier_1.train_step(data)
    cl2_results = self.classifier_2.train_step(data)

    results = {}
    for score in cl1_results:
      results["cl1_%s" % score] = cl1_results[score]
      results["cl2_%s" % score] = cl2_results[score]
    return results

  def maximize_discrepancy(self, images):
    """ Maximizes target prediction dixrepancy by training heads.
    """
    with tf.GradientTape() as tape:
      features = self.headless(images)
      pred_1 = self.head_1(features[self.target_domain_idx])
      pred_2 = self.head_2(features[self.target_domain_idx])
      discrepancy = self.discrepancy_loss(pred_1, pred_2)
      total_loss = -1. * self.mm_loss_weight * discrepancy

    trainable_vars = (self.head_1.trainable_variables +
                      self.head_2.trainable_variables)
    gradients = tape.gradient(total_loss, trainable_vars)
    self.max_discr_optimizer.apply_gradients(zip(gradients, trainable_vars))
    return discrepancy

  def minimize_discrepancy(self, images):
    """ Minimizes target prediction dixrepancy by training mixture weights.
    """
    with tf.GradientTape() as tape:
      features = self.headless(images)
      pred_1 = self.head_1(features[self.target_domain_idx])
      pred_2 = self.head_2(features[self.target_domain_idx])
      discrepancy = self.discrepancy_loss(pred_1, pred_2)
      total_loss = self.mm_loss_weight * discrepancy

    trainable_vars = self.da_trainable_parameters
    gradients = tape.gradient(total_loss, trainable_vars)
    self.min_discr_optimizer.apply_gradients(zip(gradients, trainable_vars))
    return discrepancy


  def train_step(self, data):
    images, _ = data
    # train the classifier on source domains
    out_results = {}
    if self.train_classifier:
      out_results = self.classifier_train_step(data)
    
    # min-max the target prediction discrepancy
    max_discr = self.maximize_discrepancy(images)
    min_discr = self.minimize_discrepancy(images)
    out_results.update({
        "max_discrepancy": max_discr,
        "min_discrepancy": min_discr})
    # minimize moments distance
    mm_results = self.match_moments(images)
    out_results.update(mm_results)
    return out_results

  def test_step(self, data):
    cl1_res = self.classifier_1.test_step(data)
    cl2_res = self.classifier_2.test_step(data)
    result = {}
    for score in cl1_res:
      result["cl1_%s" % score] = cl1_res[score]
      result["cl2_%s" % score] = cl2_res[score]
    return result

  def evaluate(self, *args, **kwargs):
    """ Classifiers evaluation."""
    cl1_res = self.classifier_1.evaluate(*args, **kwargs)
    cl2_res = self.classifier_2.evaluate(*args, **kwargs)
    n_res = len(cl1_res)
    if isinstance(cl1_res, dict):
      result = {}
      for score in cl1_res:
        result["cl1_%s" % score] = cl1_res[score]
        result["cl2_%s" % score] = cl2_res[score]
      return result
    # if the result is a list
    result = [0.5 * (cl1_res[i] + cl2_res[i]) for i in range(n_res)]
    return result
