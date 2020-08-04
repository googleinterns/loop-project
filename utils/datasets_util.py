import numpy as np
import os
import os.path as op
import glob
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds


USPS_PATH = "/home/dbash_google_com/datasets/usps.h5"

def hdf5(path, data_key="data", target_key="target"):
  """
      loads data from hdf5:
      - hdf5 should have 'train' and 'test' groups
      - each group should have 'data' and 'target' dataset or spcify the key
      - flatten means to flatten images N * (C * H * W) as N * D array
  """
  outputs = []
  with h5py.File(path, "r") as hf:
    for ds in ["train", "test"]:
      split = hf.get(ds)
      x = train.get(data_key)[:]
      y = train.get(target_key)[:]
      x = x.reshape((x.shape[0], 16, 16, 1))*255.
      outputs.extend(x, y)
    
  return tuple(outputs)

def get_usps_tf_dataset(path):
  """returns tf.data.Dataset object with USPS dataset.

  Arguments:
  path: path to h5 file.
  """

  x_train, y_train, x_test, y_test = hdf5(path)
  x_train, y_train = tf.constant(x_train), tf.constant(y_train)
  x_test, y_test = tf.constant(x_test), tf.constant(y_test)

  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

  return train_dataset, test_dataset


def get_split_dataset(name, data_dir, split=0.8):
  """Returns tf.data.Dataset objects for the quickdraw dataset,

  Arguments:
  name: dataset name.
  data_dir: the dataset files will be stored in this folder.
  split: training split."""
  percent = int(split*100)
  split_arg = ["train[:%i%%]" % percent, "train[%i%%:]" % percent]
  datasets, info = tfds.load(name, data_dir=data_dir,
                             with_info=True,
                             as_supervised=True,
                             split=split_arg)

  return datasets, info

def get_dataset_from_directory(dir_path, split):
  """Creates a dataset instance given a dataset directory.

  Arguments:
  dir_path: path to the dataset files.
  split: test split."""
  def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, op.sep)
    # The second to last is the class-directory
    one_hot = tf.cast(parts[-2] == classes, tf.float32)
    # Integer encode the label
    return tf.argmax(one_hot)

  def process_path(file_path):
    label = get_label(file_path)
    data = tf.io.read_file(file_path)
    image = tf.cond(
        tf.image.is_jpeg(data),
        lambda: tf.image.decode_jpeg(data, channels=3),
        lambda: tf.image.decode_png(data, channels=3))
    return image, label

  classes = [x for x in sorted(os.listdir(dir_path))
             if op.isdir(op.join(dir_path, x))]
  classes = np.array(classes)
  img_files = glob.glob(op.join(dir_path, "*/*"))
  list_ds = tf.data.Dataset.list_files(img_files, shuffle=False)
  labeled_ds = list_ds.map(process_path)
  num_files = len(img_files)
  num_test_files = int(num_files * split)
  labeled_ds = labeled_ds.shuffle(num_files, seed=123,
                                  reshuffle_each_iteration=False)
  train_ds = labeled_ds.skip(num_test_files)
  test_ds = labeled_ds.take(num_test_files)
  return train_ds, test_ds, classes

def get_office_datasets(data_path, split=0.2):
  """Returns a dictionary of Office datasets.

  Arguments:
  data_path: path to the Office dataset folder.
  split: test split."""
  domains = [f for f in os.listdir(data_path)
             if op.isdir(op.join(data_path, f))]
  datasets_dict = {}
  for domain in domains:
    domain_folder = op.join(data_path, domain, "images")
    train_ds, test_ds, classes = get_dataset_from_directory(
        domain_folder, split)
    datasets_dict[domain] = (train_ds, test_ds, classes)
  return datasets_dict

def get_domain_net_datasets(data_path, split=0.2):
  """Returns a dictionary of DomainNet datasets.

  Arguments:
  data_path: path to the DomainNet dataset folder.
  split: test split."""
  domains = [f for f in os.listdir(data_path)
             if op.isdir(op.join(data_path, f))]
  datasets_dict = {}
  for domain in domains:
    domain_folder = op.join(data_path, domain)
    train_ds, test_ds, classes = get_dataset_from_directory(
        domain_folder, split)
    datasets_dict[domain] = (train_ds, test_ds, classes)
  return datasets_dict


def preprocess_dataset(dataset, image_size, num_classes, augment=False,
                       repeat=False):
  """Preprocesses the dataset.

   Reshapes, splits into images and labels, optionally performs augmentation.

  Arguments:
  dataset: tf.data.Dataset object.
  image_size: size of the output image (integer or tuple).
  num_classes: number of classes in the dataset.
  augment: if True, augmentation will be preformed.
  repeat: if True, the dataset will be repeated."""
  def take_x(x, y):
    return x

  def take_y(x, y):
    return y

  def augment_fn(image, label):
    """ Data augmentation function."""
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 5, 10)
    return image, label

  def preprocess(x, y):
    """Reads the image and converts the label to one-hot vector."""
    x = tf.cast(x, tf.float32) / 255.
    #x = tf.image.per_image_standardization(x)
    x = tf.image.resize(x, image_size)
    if x.shape[-1] == 1:
      x = tf.repeat(x, 3, axis=-1)
    y = tf.one_hot(y, num_classes)
    return x, y

  if isinstance(image_size, int):
    image_size = (image_size, image_size)

  dataset = dataset.map(preprocess)
  if repeat:
    dataset = dataset.repeat()
  if augment:
    dataset = dataset.map(augment_fn)
  ds_x, ds_y = dataset.map(take_x), dataset.map(take_y)
  return  ds_x, ds_y


def get_dataset(name, image_size, data_dir=None, augment=False):
  """A method that returns a dictionary of the dataset content.

  Arguments:
    name: name of the dataset (needs to be available for tfds.load).
    image_size: output image size.
    data_dir: directory where the dataset files are saved.
    augment: if True, data augmentation will be used for training set.
  """

  if "usps" in name:
    train_ds, test_ds = get_usps_tf_dataset(USPS_PATH)
    num_classes = 10
    get_label_name = None
  elif "quickdraw_bitmap" in name:
    datasets, info = get_split_dataset(name="quickdraw_bitmap",
                                       split=0.8, data_dir=data_dir)
    train_ds, test_ds = datasets
    num_classes = info.features["label"].num_classes
    get_label_name = info.features["label"].int2str
  else:
    datasets, info = tfds.load(name, data_dir=data_dir,
                               with_info=True,
                               as_supervised=True)
    num_classes = info.features["label"].num_classes
    get_label_name = info.features["label"].int2str
    test_key = "test"
    if name in ["imagewang", "imagenette"]:
      test_key = "validation"
    train_ds, test_ds = datasets["train"], datasets[test_key]

  train_x, train_y = preprocess_dataset(train_ds, image_size, num_classes,
                                        augment=augment, repeat=True)
  test_x, test_y = preprocess_dataset(test_ds, image_size, num_classes,
                                      augment=False, repeat=False)
  datasets = {
      "train_x": train_x,
      "train_y": train_y,
      "test_x": test_x,
      "test_y": test_y}
  datasets_info = {"num_classes": num_classes, "get_label_name": get_label_name}
  return datasets, datasets_info


def get_da_datasets(name, image_size, data_dir, split=0.2, augment=False):
  """Returns a dictionary of a given DA dataset.

  Arguments:
  name: name of the DA dataset.
  data_path: path to the dataset folder.
  image_size: size of the output images.
  split: validation split."""

  if "office" in name:
    datasets_dict = get_office_datasets(op.join(data_dir, "office"), split)
  elif "domain_net" in name:
    datasets_dict = get_domain_net_datasets(op.join(data_dir, "domain_net"), split)
  else:
    raise ValueError("Given dataset type is not supported")

  domains = {}
  domain_info = {}
  print(datasets_dict.keys())
  for domain in datasets_dict:
    train_ds, test_ds, classes = datasets_dict[domain]
    num_classes = len(classes)

    train_x, train_y = preprocess_dataset(train_ds, image_size, num_classes,
                                          augment, repeat=True)
    test_x, test_y = preprocess_dataset(test_ds, image_size, num_classes,
                                        augment=False, repeat=False)
    domains[domain] = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y}
    domain_info[domain] = {"num_classes": num_classes,
                           "get_label_name": classes}
  return domains, domain_info
