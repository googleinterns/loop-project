"""Discriminative domain adaptation module.
"""
import tensorflow as tf
import tensorflow.keras as tkf
from lib.uda_model import BaseDomainAdaptationModel

def get_discriminator(input_shape, num_layers=3, activation="relu", dropout=0):
  """Returns the disctiminator model.

  Arguments:
  input_shape: shape of the input tensor.
  num_layers: number of convolutional layers.
  activaiton: activation function (default is relu).
  dropout: dropout.
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

class AdversarialClassifier(BaseDomainAdaptationModel):
  """Discriminative classifier for unsupervised domain adaptation.

  Consists of a classifier network that is trained to perform classification on
  the source domain, and domain disriminator that, given late features from the
  classifier, predicts whether the features come from source or target domain.
  The target domain mixture weights are optimized to fool the discriminator.
  During first epochs, only the classifier is trained to achieve better
  adaptation results.

  Arguments:
  classifier: classifier model.
  discriminator: discriminator model.
  domains: list of domain names.
  target_domain: target domain name.
  num_layers: number of resblock layers in the classifier.
  """
  def __init__(self, discriminator, *args, **kwargs):
    super(AdversarialClassifier, self).__init__(*args, **kwargs)
    self.discriminator = discriminator
    self.copy_mix_weights()

  def compile(self, d_optimizer, adv_optimizer, loss_fn, d_loss_weight=1,
              adv_loss_weight=1, *args, **kwargs):
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
    super(AdversarialClassifier, self).compile(*args, **kwargs)
    self.classifier.compile(optimizer=self.cl_optimizer,
                            loss=self.cl_losses,
                            loss_weights=self.cl_loss_weights,
                            metrics="acc")
    self.d_optimizer = d_optimizer
    self.adv_optimizer = adv_optimizer
    self.loss_fn = loss_fn
    self.d_loss_weight = d_loss_weight
    self.adv_loss_weight = adv_loss_weight


  def train_step(self, data):
    images, labels = data
    batch_size = tf.shape(labels[0])[0]

    cl_results = self.classifier.train_step(data)
    result = cl_results
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
      d_loss_weighted = (self.d_loss_weight * 0.5 *
                         (d_loss_source + d_loss_target))
    grads = tape.gradient(d_loss_weighted,
                          self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(
        zip(grads, self.discriminator.trainable_weights))
    result["d_loss_source"] = d_loss_source
    result["d_loss_target"] = d_loss_target

    # Assemble labels that say "all source images"
    misleading_labels = tf.zeros(label_shape)
    # Train the mixture weights to fool the discriminatoer
    with tf.GradientTape() as tape:
      features = self.late_features(images)
      predictions = self.discriminator(features[self.target_domain_idx])
      adv_loss = self.loss_fn(misleading_labels, predictions)
      adv_loss_weighted = (self.adv_loss_weight *
                           self.loss_fn(misleading_labels, predictions))
    grads = tape.gradient(adv_loss_weighted,
                          self.late_features._target_mix_weights)
    self.adv_optimizer.apply_gradients(
        zip(grads, self.late_features._target_mix_weights))
    result["adv_loss"] = adv_loss

    return result
