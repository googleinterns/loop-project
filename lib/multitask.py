""" Main code for multitask and domain adaptation experiments.
"""
import os
import tensorflow as tf
import tensorflow.keras as tkf
import lib.weighted_resblock as wblock
import lib.shared_resnet as sresnet
import lib.custom_resnet as cresnet
from utils import args_util, training, datasets_util
from lib.resnet_parameters import ResNetParameters

# dictionary of lists of datasets
DATASET_TYPE_DICT = {
    "digits_uda": ["mnist", "mnist_corrupted/shot_noise",
                   "mnist_corrupted/shear",
                   "mnist_corrupted/scale",
                   "svhn_cropped",
                   "usps"],
    "digits_da": ["mnist", "mnist_corrupted/scale",
                  "mnist_corrupted/glass_blur", "usps", "svhn_cropped"],
    "digits": ["mnist_corrupted/glass_blur",
               "mnist_corrupted/scale",
               "usps"],
    "characters": ["kmnist", "emnist", "omniglot",
                   "quickdraw_bitmap", "cmaterdb"],
    "small_natural": ["cifar100", "imagenette",
                      "cifar10"],
    "cifar10": ["cifar10_corrupted/zoom_blur_1",
                "cifar10_corrupted/shot_noise_2",
                "cifar10_corrupted/fog_3",
                "cifar10_corrupted/gaussian_blur_2",
                "cifar10_corrupted/zoom_blur_5",
                "cifar10"],
    "natural": ["imagenette", "caltech101", "imagenet2012",
                "oxford_iiit_pet", "pet_finder"],
    "mix": ["cifar100", "quickdraw_bitmap", "mnist",
            "imagewang", "imagenette", "usps",
            "svhn_cropped", "kmnist", "cifar10"],
    "mix_2": ["cifar100", "mnist", "imagenette",
              "usps", "svhn_cropped", "kmnist"],
    "office": "office",
    "domain_net": "domain_net",
    "domain_net_small": "domain_net_small",
    "domain_net_tiny": "domain_net_tiny",
    "domain_net_subset": "domain_net_subset",
    "domain_net_subset_augmented": "domain_net_subset_augmented"}


def get_custom_parser():
  """returns a customized arguments parser."""
  parser = args_util.get_parser()
  parser.add_argument("--num_datasets",
                      type=int,
                      default=1,
                      help="number of datasets in pretraining phase.")
  parser.add_argument("--dataset_type",
                      choices=list(DATASET_TYPE_DICT.keys()),
                      default="digits",
                      help="dataset type (default is digits)")
  parser.add_argument("--share_mixture",
                      type=int,
                      default=0,
                      help="if > 0, the mixture weights are shared.")
  parser.add_argument("--share_logits",
                      type=int,
                      default=0,
                      help="if > 0, the last layer is shared across datasets.")
  parser.add_argument("--copy_weights",
                      type=int,
                      default=0,
                      help="if > 0, mixture weights and"
                           "head will be shared to target.")
  parser.add_argument("--num_epochs_finetune",
                      type=int,
                      default=10,
                      help="number of epochs to finetune the model.")
  parser.add_argument("--finetune_mode",
                      choices=["h", "m", "b", "bm", "hm", "hb", "hbm", "all"],
                      default="h",
                      help="finetuning mode "
                      "(h = head, m = mixture weights,"
                      "b = batch norm, all = entire model.)")
  return parser

def write_scores(datasets, scores, fpath, f_mode="w+",
                 scores_name="Pretraining"):
  """Writes the evaluation scores to the file.
  """
  num_dsets = len(datasets)
  fl = open(fpath, f_mode)
  fl.write("%s results\n" % scores_name)
  for i in range(num_dsets):
    print("Test loss on %s: %f" % (datasets[i], scores[i + 1]))
    print("Test accuracy on %s: %f" % (datasets[i], scores[num_dsets + i + 1]))
    fl.write("Test loss on %s: %f\n" % (datasets[i], scores[i + 1]))
    fl.write("Test accuracy on %s: %f\n\n" %
             (datasets[i], scores[num_dsets + i + 1]))
  fl.close()

def get_losses(datasets, num_train_ds, label_smoothing=0):
  """Returns dictionaries of losses and loss weights for the datasets.

  Arguments:
  datasets: list of dataset names.
  num_train_ds: number of datasets with non-zero loss weight.
  label_smoothing: label smoothing coefficient.
  """
  cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
  loss_weights = {}
  losses = {}
  for i in range(len(datasets)):
    loss_name = "%s_out" % datasets[i]
    loss_weights[loss_name] = float(i < num_train_ds)
    losses[loss_name] = cce
  return losses, loss_weights

def get_losses_and_callbacks(
    datasets, num_tr_datasets, prefix, add_mw_callback=False,
    train_params=training.TrainingParameters(), end_lr_coefficient=0.1):
  lr_schedule = training.get_lr_schedule(
      train_params.lr, train_params.num_epochs,
      end_lr_coefficient=end_lr_coefficient)
  losses, loss_weights = get_losses(
      datasets, num_tr_datasets, train_params.lsmooth)
  print("Loss weights:", loss_weights)
  callbacks, ckpt = training.get_callbacks(
      train_params.save_path, lr_schedule, prefix)
  if add_mw_callback:
    vis_path = os.path.join(train_params.save_path, "mix_vis", prefix)
    vis_cbk = training.VisualizeCallback(
        vis_path, domains=datasets, num_templates=train_params.num_templates,
        num_layers=train_params.num_layers, frequency=1)
    callbacks.append(vis_cbk)
  return callbacks, lr_schedule, losses, loss_weights, ckpt


def _get_pretrain_datasets(datasets, image_size, data_dir, augment):
  """A method that returns a dictionary pretraining datasets.

  Arguments:
    datasets_list: list of datasets names (need to be available for tfds.load).
    image_size: output image size.
    data_dir: directory where the dataset files are saved.
    augment: if True, data augmentation will be used for training set.
  """
  if "office" in datasets or "domain_net" in datasets:
    return datasets_util.get_da_datasets(datasets, image_size, data_dir,
                                         split=0.2, augment=augment)

  datasets_dict = {}
  datasets_info = {}
  for ds in datasets:
    dataset, info = datasets_util.get_dataset(ds, image_size, data_dir, augment)
    datasets_dict[ds] = dataset
    datasets_info[ds] = info
  return datasets_dict, datasets_info


def _combine_datasets(datasets_dict):
  """Returns train and test tf.data.Dataset objects containing all datasets.

  Arguments:
  datasets_dict: dictionary containing the datasets (to be acquired with
  `get_datasets` method).
  """
  train_x_datasets = [datasets_dict[x]["train_x"] for x in datasets_dict]
  train_y_datasets = [datasets_dict[x]["train_y"] for x in datasets_dict]
  test_x_datasets = [datasets_dict[x]["test_x"] for x in datasets_dict]
  test_y_datasets = [datasets_dict[x]["test_y"] for x in datasets_dict]

  combined_dataset_train_x = tf.data.Dataset.zip(tuple(train_x_datasets))
  combined_dataset_train_y = tf.data.Dataset.zip(tuple(train_y_datasets))
  combined_dataset_test_x = tf.data.Dataset.zip(tuple(test_x_datasets))
  combined_dataset_test_y = tf.data.Dataset.zip(tuple(test_y_datasets))

  combined_dataset_train = tf.data.Dataset.zip((combined_dataset_train_x,
                                                combined_dataset_train_y))
  combined_dataset_test = tf.data.Dataset.zip((combined_dataset_test_x,
                                               combined_dataset_test_y))
  return combined_dataset_train, combined_dataset_test

def get_datasets(datasets, new_shape,
                 data_dir, augment=False, batch_size=32):
  """Returns train and test data objects along with ifo.

  Arguments:
    datasets: list of dataset names.
    new_shape: the images be reshape to this shape.
    data_dir: directory where the datasets are stored.
    augment: if True, images will be augmented.
    batch_size: batch size.
  """
  dataset_dict, info = _get_pretrain_datasets(
      datasets, image_size=new_shape,
      data_dir=data_dir, augment=augment)

  ds_train, ds_test = _combine_datasets(dataset_dict)
  ds_train = (ds_train.shuffle(buffer_size=100)
              .batch(batch_size, drop_remainder=True))
  ds_test = (ds_test.shuffle(buffer_size=100)
             .batch(batch_size, drop_remainder=True))
  return ds_train, ds_test, info

def build_model(feature_extractor, datasets, input_shape,
                share_logits=False, with_head=True):
  """Returns combined model for the given datasets.

  Creates a tf.keras.Model that has separate inputs and top layers for all
  datasets in the experiment and shared feature extractor and I/O adapters.

  Arguments:
  feature_extractor: common feature extractor without head.
  datasets: dataset dict (from `get_datasets` method).
  input_shape: model input shape.
  share_logits: If True, single logits layer will be shared across datasets,
  otherwise separate logits layers will be created for each dataset.
  """
  def rename(inputs, name):
    return tkf.layers.Lambda(lambda x: x, name=name)(inputs)

  num_classes = [datasets[x]["num_classes"] for x in datasets]
  ds_names = list(datasets.keys())
  num_datasets = len(datasets)

  inputs = [tkf.Input(name="%s_in" % x, shape=input_shape)
            for x in ds_names]
  features = [feature_extractor(x) for x in inputs]
  if with_head:
    if share_logits:
      out_layer = tkf.layers.Dense(num_classes[0], activation="softmax",
                                   name="shared_out")
      outputs = [rename(out_layer(features[i]), "%s_out" % ds_names[i])
                 for i in range(num_datasets)]
    else:
      outputs = [tkf.layers.Dense(num_classes[i], activation="softmax",
                                  name="%s_out" % ds_names[i])(features[i])
                 for i in range(num_datasets)]
  else:
    outputs = features
  combined_model = tkf.Model(inputs, outputs, name="combined")
  return combined_model


def build_mixture_model(feature_extractor, datasets, input_shape,
                        num_layers, num_templates,
                        with_head=True,
                        share_logits=False):
  """Returns combined mixture weights model for the given datasets.

  Creates a tf.keras.Model that has separate inputs and top layers for all
  datasets in the experiment and shared feature extractor and I/O adapters.

  Arguments:
  feature_extractor: common feature extractor without head.
  datasets: dataset dict (from `get_datasets` method).
  input_shape: model input shape.
  num_layers: number of residual layers in feature extractor.
  num_templates: number of templates for mixture weights.
  share_logits: If True, single logits layer will be shared across datasets,
  otherwise separate logits layers will be created for each dataset.
  """
  def get_mixture_weights(prefix, x):
    mixture_weights = []
    for i in range(num_layers):
      name = "%s_mix_%s" % (prefix, str(i))
      mix_weight = wblock.MixtureWeight(num_templates=num_templates,
                                        name=name)(x)
      mixture_weights.append(mix_weight)
    return mixture_weights

  def get_output_layer(num_classes, prefix):
    return tkf.layers.Dense(num_classes, activation="softmax",
                            name="%s_out" % prefix)
  def rename(inputs, name):
    return tkf.layers.Lambda(lambda x: x, name=name)(inputs)

  num_classes = [datasets[x]["num_classes"] for x in datasets]
  ds_names = list(datasets.keys())
  num_datasets = len(datasets)

  inputs = [tkf.Input(name="%s_in" % x, shape=input_shape) for x in ds_names]
  mix_weights = [get_mixture_weights(name, inputs[0]) for name in ds_names]
  features = [feature_extractor([inputs[i], *mix_weights[i]])
              for i in range(num_datasets)]
  if with_head:
    if share_logits:
      out_layer = get_output_layer(num_classes[0], "shared")
      outputs = [rename(out_layer(features[i]), "%s_out" % ds_names[i])
                 for i in range(num_datasets)]
    else:
      out_layers = [get_output_layer(num_classes[i], ds_names[i])
                    for i in range(num_datasets)]
      outputs = [out_layers[i](features[i]) for i in range(num_datasets)]
  else:
    outputs = features

  combined_model = tkf.Model(inputs, outputs, name="combined")
  return combined_model

def get_combined_model(
    datasets_info, model_params: ResNetParameters,
    shared=False, share_logits=False, with_head=True):
  """Returns the multitask training model.

  Arguments:
  datasets_info: dataset info dictionary.
  shared: if True, the shared model will be created.
  model_params: feature extractor parameters (ResNetParameters).
  share_logits: if True, the logits layer will be shared.
  """
  model_params.name = "feature_extractor"
  model_params.mixture_weights_as_input = True
  if not shared:
    feature_extractor = cresnet.resnet(model_params)
    combined_model = build_model(
        feature_extractor=feature_extractor, datasets=datasets_info,
        input_shape=model_params.input_shape, share_logits=share_logits,
        with_head=with_head)
  else:
    feature_extractor = sresnet.shared_resnet(model_params)
    combined_model = build_mixture_model(
        feature_extractor=feature_extractor, datasets=datasets_info,
        input_shape=model_params.input_shape,
        num_layers=model_params.num_layers,
        num_templates=model_params.num_templates,
        share_logits=share_logits, with_head=with_head)
  return combined_model


def train_model(model, train_data, test_data, datasets,
                num_train_datasets, target_dataset=None,
                finetune_mode="all", shared=False, prefix="Pretrained",
                train_params=training.TrainingParameters()):
  """Pretrains or finetunes the model on source/target datasets.

  Arguments:
  model: combined model.
  train_data: training data.
  test_data: test data.
  datasets: list of datasets names.
  num_train_datasets: number of datasets to thain the model on.
  target_dataset: target dataset. If not None, only the target loss will be
  minimized, and the weights will be fixed.
  train_params: training parameters (TrainingParameters object).
  finetune_mode: string representing the finetuning mode.
  shared: if True, it is assumed that the model is shared.
  """
  fitting_info = get_losses_and_callbacks(
      datasets=datasets, num_tr_datasets=num_train_datasets, prefix=prefix,
      end_lr_coefficient=0.2, train_params=train_params, add_mw_callback=shared)
  callbacks, lr_schedule, losses, loss_weights, ckpt = fitting_info
  if train_params.restore:
    training.restore_model(ckpt, model)
  # modify loss weights if target_dataset is given
  if target_dataset is not None:
    for loss in loss_weights:
      loss_weights[loss] = float(("%s_out" % target_dataset) == loss)
    fixed_model = fix_weights(model, target_dataset=target_dataset,
                              finetune_mode=finetune_mode,
                              shared=shared)
    fixed_model.summary()
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0))

  # fitting the model
  trained_model = _fit_model(
      model, train_data, test_data, callbacks=callbacks, optimizer=optimizer,
      losses=losses, loss_weights=loss_weights,
      num_epochs=train_params.num_epochs, num_steps=train_params.num_steps,
      batch_size=train_params.batch_size)
  # saving the model
  trained_model.save_weights(ckpt)
  return trained_model


def _fit_model(model, train_data, test_data, callbacks, optimizer,
               losses, loss_weights, num_epochs=100, num_steps=None,
               batch_size=32):
  """Compiles and fits the model to the training data.

  Arguments:
  model: tf.keras.Model object.
  train_data: tf.data.Dataset training set object.
  test_data: tf.data.Dataset test set object.
  callbacks: callbacks.
  optimizer: tf.keras.optimizers object.
  losses: dictionary of losses.
  loss_weights: dictionary of loss weights.
  num_epochs: number of epochs to fit.
  num_steps: number of steps in one epoch.
  batch_size: batch_size.
  """
  model.compile(loss=losses, optimizer=optimizer, metrics=["acc"],
                loss_weights=loss_weights)

  # training the model
  if num_epochs > 0:
    model.fit(train_data.prefetch(tf.data.experimental.AUTOTUNE),
              verbose=1, batch_size=batch_size, epochs=num_epochs,
              steps_per_epoch=num_steps, validation_data=test_data,
              callbacks=callbacks)
  return model


def fix_weights(model, target_dataset, finetune_mode="h", shared=False):
  """Fixes model weights according to experiment type.

  Arguments:
  model: the tf.keras.Model object.
  target_dataset: string target dataset name.
  finetune_mode: a string combination of  the following characters that
    tells which variables to set trainable.
    Options:`h` - model head, `m` - mixture weights,
    `b` - batch norm weights, or `all` - all variables.
  shared: whether the model uses shared templates.
  """
  trainable_names = []
  if finetune_mode == "all":
    print("All model weights are trainable.")
    return model
  else:
    if "h" in finetune_mode:
      target_head_name = target_dataset + "_out"
      trainable_names.append(target_head_name)
      trainable_names.append("shared_out")
    if "m" in finetune_mode:
      trainable_names.append("shared_mix")
      trainable_names.append(target_dataset + "_mix")
  for layer in model.layers:
    if any([name in layer.name for name in trainable_names]):
      print("Layer set to be trainable: ", layer.name)
      layer.trainable = True
    else:
      layer.trainable = False

  if "b" in finetune_mode:
    if shared:
      f_extr = model.get_layer("feature_extractor")
      f_extr.trainable = True
      for layer in f_extr.layers:
        if "weighted_res_block_separate_bn" in layer.name:
          resblock = layer.resblock
          layer.trainable = True
        elif "weighted_resblock" in layer.name:
          resblock = layer
          layer.trainable = True
        else:
          layer.trainable = False

      for layer in resblock.layers:
        if "weighted_batch_normalization" in layer.name:
          print("Layer set to be trainable: ", layer.name)
          layer.trainable = True
        else:
          layer.trainable = False
    else:
      feature_extr = model.get_layer("feature_extractor")
      feature_extr.trainable = True
      for layer in feature_extr.layers:
        if "res_block" in layer.name:
          layer.trainable = True
          for block_layer in layer.layers:
            if "batch_normalization" in block_layer.name:
              print("Layer set to be trainable: ", block_layer.name)
              block_layer.trainable = True
            else:
              block_layer.trainable = False
        else:
          layer.trainable = False
  print(model.trainable_variables)
  return model

def copy_weights(model, source, target, num_layers, shared=False):
  """Copies the weight values of mixture weights and head from source to target
  domain.

  Arguments:
  model: a tf.keras.Model object.
  source: source domain name.
  target: target domain name.
  shared: whether the model is shared.
  num_layers: number of resblock layers in the model.
  """
  try:
    model.get_layer("shared_out")
  except ValueError:
    try:
      source_out = model.get_layer("%s_out" % source)
      target_out = model.get_layer("%s_out" % target)
      target_out.set_weights(source_out.get_weights())
      print("copied head weights.")
    except ValueError:
      print("No head to copy.")

  if shared:
    for idx in range(num_layers):
      source_mix = model.get_layer("%s_mix_%d" % (source, idx))
      target_mix = model.get_layer("%s_mix_%d" % (target, idx))
      target_mix.set_weights(source_mix.get_weights())
      print("copied weights from %s_mix_%d to %s_mix_%d" %
            (source, idx, target, idx))

  return model
