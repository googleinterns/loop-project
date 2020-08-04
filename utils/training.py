import os
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
      update_freq=1500, profile_batch=10, embeddings_freq=0,
      embeddings_metadata=None)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1,
                                                   save_freq=10000)
  #lrtboard_callback = LRTensorBoard(log_dir=log_path)
  callbacks = [cp_callback, lr_reducer, lr_scheduler, tboard]
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
    equal to initial_lrate * (1 - end_lr_coefficient)"""
  decay = end_lr_coefficient / (1.0 * num_epochs)
  def lr_schedule(epoch):
    return initial_lrate * (1 - epoch * decay)

  return lr_schedule

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  """Custom callbacks class for learning rate plotting in Tenforboard."""
  def __init__(self, log_dir, **kwargs):
    super().__init__(log_dir=log_dir, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    """Updates learning rate log at the end of epoch."""
    logs = logs or {}
    logs.update({"lr": K.eval(self.model.optimizer.lr)})
    super().on_epoch_end(epoch, logs)

