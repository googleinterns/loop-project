import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as tkf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from utils import usps, args_util

USPS_PATH = "/home/dbash_google_com/datasets/usps.h5"

def get_callbacks(save_path, lr_schedule, prefix=None):
  if prefix is None:
    prefix = "checkpoint"
  log_path = os.path.join(save_path,"logs", prefix)
  checkpoint_path = os.path.join(save_path, "checkpoints",
                                 "%s.ckpt" % prefix)

  lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0,
    patience=5, min_lr=0.5e-6)
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
  tboard = tf.keras.callbacks.TensorBoard(
      log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
      update_freq=1500, profile_batch=0, embeddings_freq=0,
      embeddings_metadata=None)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  save_freq=10000)
  lrtboard_callback = LRTensorBoard(log_dir=log_path)
  callbacks = [cp_callback, lr_reducer, lr_scheduler, tboard, lrtboard_callback]
  return callbacks, checkpoint_path


def get_lr_schedule(initial_lrate, num_epochs):
  """returns a learning rate schedule function."""
  decay = 0.95 * initial_lrate / (1.0 * num_epochs)
  def lr_schedule(epoch):
    return initial_lrate - epoch * decay
  
  return lr_schedule

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  """Custom callbacks class for learning rate plotting in Tenforboard."""
  def __init__(self, log_dir, **kwargs):
    super().__init__(log_dir=log_dir, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    """Updates learning rate log at the end of epoch."""
    logs = logs or {}
    logs.update({'lr': K.eval(self.model.optimizer.lr)})
    super().on_epoch_end(epoch, logs)
    

def get_dataset(name, image_size, data_dir=None, augment=False):
  """A method that returns a dictionary of the dataset content.
  
  Arguments:
    name: name of the dataset (needs to be available for tfds.load).
    image_size: output image size.
    data_dir: directory where the dataset files are saved.
    augment: if True, data augmentation will be used for training set.
  """

  def augment(image, label):
    """ Data augmentation function."""
    image = tf.image.random_brightness(image, max_delta=0.3) # Random brightness
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 5, 10)
    return image, label

  def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.
    #x = tf.image.per_image_standardization(x)
    x = tf.image.resize(x, image_size)
    if x.shape[-1] == 1:
      x = tf.repeat(x, 3, axis=-1)
    y_cat = tf.one_hot(y, num_classes)
    return x, y_cat

  def take_x(x, y):
    return x

  def take_y(x, y):
    return y

  if name is "usps":
    train_ds, test_ds = usps.get_usps_tf_dataset(USPS_PATH)
    num_classes = 10
    get_label_name = None
  else:
    datasets, info = tfds.load(name, data_dir=data_dir,
                                with_info=True, 
                                as_supervised=True)
    num_classes = info.features["label"].num_classes
    get_label_name = info.features['label'].int2str
    train_ds, test_ds  = datasets["train"], datasets["test"]

  train_ds = train_ds.map(preprocess).repeat()  
  if augment:
    train_ds = train_ds.map(augment)
  train_x, train_y = train_ds.map(take_x), train_ds.map(take_y)
  test_ds =test_ds.map(preprocess)
  test_x, test_y = test_ds.map(take_x), test_ds.map(take_y)
  datasets_content = {
      "train_x": train_x,
      "train_y": train_y,
      "test_x": test_x,
      "test_y": test_y,
      "num_classes": num_classes,
      "get_label_name": get_label_name}
  return datasets_content
