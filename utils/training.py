import os
from dataclasses import dataclass
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def get_callbacks(save_path, lr_schedule, prefix=None):
  """ Creates callbacks.

  Arguments:
  save_path: the logs and checkpoints will be stored here.
  lr_schedule: learning rate schedule.
  prefix: prefix for the file names (default is `checkpoints`)"""
  if prefix is None:
    prefix = "checkpoint"
  log_path = os.path.join(save_path, "logs", prefix)
  checkpoint_path = os.path.join(save_path, "checkpoints",
                                 "%s.ckpt" % prefix)

  lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
      factor=np.sqrt(0.1), cooldown=0,
      patience=5, min_lr=0.5e-6)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
  tboard = tf.keras.callbacks.TensorBoard(
      log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
      update_freq=1500, profile_batch="10,20", embeddings_freq=0,
      embeddings_metadata=None)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1,
                                                   save_freq=10000)
  nan_callback = tf.keras.callbacks.TerminateOnNaN()
  #lrtboard_callback = LRTensorBoard(log_dir=log_path)
  callbacks = [cp_callback, lr_reducer, lr_scheduler, tboard, nan_callback]
  return callbacks, checkpoint_path


def get_lr_schedule(initial_lrate, num_epochs, end_lr_coefficient=0.95):
  """Returns a learning rate schedule function.

  The learning rate schedule will have a linear decay described by
  the following formula:
    lr_k = initial_lrate(1 - k * end_lr_coefficient / num_epochs),
    where k is the number of epoch.

  Arguments:
  initial_lrate: initial learning rate.
  num_epochs: number of epochs.
  end_lr_coefficient: the learning rate at the final epoch will be
    equal to initial_lrate * (1 - end_lr_coefficient).
  """
  decay = end_lr_coefficient / (float(num_epochs) + 1e-6)
  def lr_schedule(epoch):
    return initial_lrate * (1 - epoch * decay)

  return lr_schedule

def visualize_mixture_weights(model, domains, num_templates, num_layers):
  """ Returns a matrix of mixture weights for visualization.

  Arguments:
  model: target model.
  domains: list of domain names.
  num_templates: number of templates in the model.
  num_layers: number of resblock layers in the model.
  """
  num_domains = len(domains)
  mix_w_matr = np.zeros((num_layers, num_domains * (num_templates + 1)))
  domain_idx = 0
  k = (num_templates + 1)
  for domain in domains:
    mix_w_arr = np.zeros((num_layers, num_templates))
    for i in range(num_layers):
      mw = model.get_layer("%s_mix_%s" % (domain, i))
      if len(mw.trainable_variables) == 0:
        mw_weights = mw.non_trainable_variables[0]
      else:
        mw_weights = mw.trainable_variables[0]
      mix_w_arr[i] = tf.nn.softmax(mw_weights, axis=1).numpy()
    mix_w_matr[:, domain_idx * k: (domain_idx + 1) * k - 1] = mix_w_arr
    domain_idx += 1

  return mix_w_matr


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  """Custom callbacks class for learning rate plotting in Tenforboard."""
  def __init__(self, log_dir, **kwargs):
    super().__init__(log_dir=log_dir, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    """Updates learning rate log at the end of epoch."""
    logs = logs or {}
    logs.update({"lr": K.eval(self.model.optimizer.lr)})
    super().on_epoch_end(epoch, logs)

class VisualizeCallback(tf.keras.callbacks.Callback):
  """
  Mixture weights visualization callback.

  Arguments:
  save_path: the mixture weight plots will be saved in this directory.
  domains: domain names.
  num_templates: number of templates.
  num_layers: number of layers.
  frequency: frequency of saving the mixture weights.
  """
  def __init__(self, save_path, domains, num_templates, num_layers,
               frequency=10, **args):
    self.frequency = frequency
    self.save_path = os.path.abspath(save_path)
    self.domains = domains
    self.num_templates = num_templates
    self.num_layers = num_layers
    # self.summary = tf.summary.create_file_writer(os.path.join(self.save_path,
    #                                                           "imglog"))
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    super(VisualizeCallback, self).__init__(**args)


  def on_epoch_end(self, epoch, logs=None):
    """Writes the mixture weight image at end of epoch.

    Arguments:
    epoch: number of epoch.
    """
    if epoch % self.frequency == 0:
      mw_img = visualize_mixture_weights(self.model, domains=self.domains,
                                         num_templates=self.num_templates,
                                         num_layers=self.num_layers)
      fname = os.path.join(self.save_path, "mixtures_%d.png" % epoch)
      imageio.imwrite(fname, mw_img)

def restore_model(ckpt_path, model):
  """ Restore model weight from the checkpoint.
  """
  try:
    model.load_weights(ckpt_path)
    print("Restored weights from %s" % ckpt_path)
  except ValueError:
    print("could not restore weights from %s" % ckpt_path)
    pass


@dataclass
class TrainingParameters:
  """ Model fitting parameters class.

  Arguments:
  num_epochs: number of epochs.
  num_steps: number of steps per epoch.
  lr: learning rate.
  lsmooth: label smoothing parameter.
  save_path: experiment files save path.
  name: experiment name.
  ckpt_path: checkpoint path.
  """
  save_path: str = "./"
  name: str = "default"
  num_epochs: int = 100
  num_steps: int = 1000
  lr: float = 2*1e-3
  lsmooth: float = 0.
  ckpt_path: str = ""
  num_layers: int = 16
  num_templates: int = 4
  batch_size: int = 32
  restore: bool = False

  def init_from_args(self, args):
    """Initializes the fields from arguments."""
    self.num_epochs = args.num_epochs
    self.num_steps = args.num_steps
    self.lsmooth = args.lsmooth
    self.lr = args.lr
    self.name = args.name
    self.save_path = args.save_path
    self.exp_path = os.path.join(self.save_path, self.name)
    self.ckpt_path = args.ckpt_path
    self.num_layers = args.num_blocks
    self.num_templates = args.num_templates
    self.batch_size = args.batch_size
    self.restore = args.restore > 0

