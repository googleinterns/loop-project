import tensorflow as tf
from lib import multitask as mt
from lib import custom_resnet as cresnet
from lib import shared_resnet as sresnet
import lib.weighted_resblock as wblock

class MultitaskTest(tf.test.TestCase):
  """Tests the multitask.py code."""
  def setUp(self):
    self.usps_path = "/home/dbash_google_com/datasets/usps.h5"
    self.data_dir = "/home/dbash_google_com/datasets/"

  def test_losses(self):
    """Tests the get_losses function."""
    datasets = ["A", "B", "C"]
    num_ds = 2
    losses, loss_weights = mt.get_losses(datasets, num_ds, label_smoothing=0)
    idx = 0
    for ds in datasets:
      expected_loss_name = "%s_out" % ds
      self.assertIn(expected_loss_name, losses)
      self.assertIn(expected_loss_name, loss_weights)
      if idx < num_ds:
        self.assertEqual(loss_weights[expected_loss_name], 1.)
      else:
        self.assertEqual(loss_weights[expected_loss_name], 0.)
      idx += 1

  def test_pretrain_datasets(self):
    """ Checks the get_pretrain_datasets function."""
    datasets_dict, _ = mt.get_pretrain_datasets("office", image_size=[32, 32],
                                                data_dir=self.data_dir,
                                                augment=False)
    ds_keys = list(datasets_dict.keys())
    self.assertAllEqual(ds_keys, ["amazon", "dslr", "webcam"])

    ds_list = ["cifar10", "mnist", "usps"]
    datasets_dict, _ = mt.get_pretrain_datasets(ds_list, image_size=[32, 32],
                                                data_dir=self.data_dir,
                                                augment=False)
    ds_keys = list(datasets_dict.keys())
    self.assertAllEqual(ds_keys, ds_list)

  def test_build_models(self):
    """Checks the build_models function."""
    input_shape = [32, 32, 3]
    datasets = {
        "mnist": {"num_classes": 10},
        "cifar100": {"num_classes": 100},
        "random": {"num_classes": 17}
    }
    feature_extractor = cresnet.resnet(
        input_shape=input_shape, with_head=False, name="feature_extractor",
        in_adapter="strided", out_adapter="isometric")
    model = mt.build_models(feature_extractor, datasets, input_shape)
    for ds in datasets:
      in_name = "%s_in" % ds
      out_name = "%s_out" % ds
      self.assertIsInstance(model.get_layer(in_name),
                            tf.keras.layers.InputLayer)
      self.assertIsInstance(model.get_layer(out_name),
                            tf.keras.layers.Dense)

  def test_build_mixture_models(self):
    """Checks the build_mixture_models function."""
    input_shape = [32, 32, 3]
    datasets = {
        "mnist": {"num_classes": 10},
        "cifar100": {"num_classes": 100},
        "random": {"num_classes": 17}}
    feature_extractor = sresnet.shared_resnet(
        input_shape=input_shape,
        in_adapter="strided", out_adapter="isometric",
        num_layers=5, num_templates=4,
        with_head=False, mixture_weights_as_input=True,
        name="feature_extractor")
    model = mt.build_mixture_models(feature_extractor, datasets, input_shape,
                                    num_layers=5, num_templates=4,
                                    use_shared_mixture_weights=False,
                                    share_logits=False)
    for ds in datasets:
      in_name = "%s_in" % ds
      out_name = "%s_out" % ds
      self.assertIsInstance(model.get_layer(in_name),
                            tf.keras.layers.InputLayer)
      self.assertIsInstance(model.get_layer(out_name),
                            tf.keras.layers.Dense)
      for layer in range(5):
        mix_name = "%s_mix_%s" % (ds, str(layer))
        self.assertIsInstance(model.get_layer(mix_name), wblock.MixtureWeight)
