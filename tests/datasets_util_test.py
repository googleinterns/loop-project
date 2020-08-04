import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import datasets_util as du

class DatasetsTest(tf.test.TestCase):
  """Tests some functions in the datasets_util."""
  def setUp(self):
    self.usps_path = "/home/dbash_google_com/datasets/usps.h5"
    self.data_dir = "/home/dbash_google_com/datasets/"

  def _check_split(self, split, train_ds, test_ds):
    """Checks if the train and test data are of the of correct proportion."""
    num_train = tf.data.experimental.cardinality(train_ds).numpy()
    num_test = tf.data.experimental.cardinality(test_ds).numpy()
    num_total = num_train + num_test
    self.assertEqual(num_train, np.ceil(split * num_total))
    self.assertEqual(num_test, np.floor((1 - split) * num_total))

  def test_usps(self):
    """Tests if get_usps_tf_dataset doesn't crash and returns
    two tf.data.Dataset objects."""
    train_ds, test_ds = du.get_usps_tf_dataset(self.usps_path)
    self.assertIsInstance(train_ds, tf.data.Dataset,
                          "tf.data.Dataset object not returned")
    self.assertIsInstance(test_ds, tf.data.Dataset,
                          "tf.data.Dataset object not returned")

  def test_split_fn(self):
    """Checks if get_split_dataset returns splits of correct proportion."""
    split = 0.6
    datasets, _ = du.get_split_dataset("quickdraw_bitmap", self.data_dir, split)
    train_ds, test_ds = datasets
    self._check_split(split, train_ds, test_ds)

  def test_office(self):
    """Checks if the get_office_datasets function returns a
    tf.data.Dataset object."""
    office_path = os.path.join(self.data_dir, "office")
    split = 0.3
    ds_dict = du.get_office_datasets(office_path, split=split)
    key_list = list(ds_dict.keys())
    self.assertAllEqual(key_list, ["amazon", "dslr", "webcam"])

    for domain in ds_dict:
      train_ds, test_ds, classes = ds_dict[domain]
      self.assertIsInstance(train_ds, tf.data.Dataset,
                            "tf.data.Dataset object not returned")
      self.assertIsInstance(test_ds, tf.data.Dataset,
                            "tf.data.Dataset object not returned")
      self.assertEqual(len(classes), 31)
      self._check_split(1. - split, train_ds, test_ds)

  def test_domain_net(self):
    """Checks if the get_domain_net_datasets function returns a
    tf.data.Dataset object."""
    domain_net_path = os.path.join(self.data_dir, "domain_net")
    split = 0.3
    ds_dict = du.get_domain_net_datasets(domain_net_path, split=split)
    key_list = list(ds_dict.keys())
    self.assertAllEqual(key_list, ["clipart", "infograph", "painting",
                                   "real", "sketch"])

    for domain in ds_dict:
      train_ds, test_ds, classes = ds_dict[domain]
      self.assertIsInstance(train_ds, tf.data.Dataset,
                            "tf.data.Dataset object not returned")
      self.assertIsInstance(test_ds, tf.data.Dataset,
                            "tf.data.Dataset object not returned")
      self.assertEqual(len(classes), 345)
      self._check_split(1. - split, train_ds, test_ds)

  def test_preprocess(self):
    """Tests if the preprocess_dataset function returns correct objects."""
    image_size = [64, 64]
    datasets, _ = tfds.load("cifar10", data_dir=self.data_dir,
                            with_info=True,
                            as_supervised=True)
    train_ds = datasets["train"]
    train_x, train_y = du.preprocess_dataset(
        train_ds, image_size, 10, augment=False, repeat=False)

    for image, label in tf.data.Dataset.zip(tuple([train_x, train_y])).take(1):
      self.assertEqual(image.shape, (64, 64, 3))
      self.assertEqual(label.shape, (10,))
