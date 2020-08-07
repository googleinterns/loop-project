""" Main code for multitask and domain adaptation experiments.
"""

import tensorflow as tf
import tensorflow.keras as tkf
import lib.weighted_resblock as wblock
import lib.shared_resnet as sresnet
import lib.custom_resnet as cresnet
from utils import args_util, training, datasets_util

# dictionary of lists of datasets
DATASET_TYPE_DICT = {
    "digits_da": ["mnist", "usps", "kmnist", "fashion_mnist",
                  "mnist_corrupted/glass_blur", "svhn_cropped"],
    "digits": ["mnist_corrupted/shot_noise",
               "mnist_corrupted/impulse_noise",
               "mnist_corrupted/glass_blur", "mnist_corrupted/shear",
               "mnist_corrupted/scale", "mnist_corrupted/fog",
               "mnist_corrupted/spatter",
               "mnist_corrupted/canny_edges",
               "mnist", "usps", "svhn_cropped"],
    "characters": ["kmnist", "emnist", "omniglot",
                   "quickdraw_bitmap", "cmaterdb"],
    "small_natural": ["cifar100", "imagenette",
                      "cifar10_corrupted/zoom_blur_1",
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
    "domain_net_tiny": "domain_net_tiny"}


def get_custom_parser():
  """returns a customized arguments parser."""
  parser = args_util.get_parser()
  parser.add_argument("--num_datasets",
                      type=int,
                      default=1,
                      help="number of datasets in pretraining phase.")
  parser.add_argument("--target_dataset",
                      type=str,
                      default="svhn_cropped",
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
  parser.add_argument("--num_epochs_finetune",
                      type=int,
                      default=10,
                      help="number of epochs to finetune the model.")
  parser.add_argument("--finetune_mode",
                      choices=["h", "m", "b", "hm", "hb", "hbm", "all"],
                      default="h",
                      help="finetuning mode "
                      "(h = head, m = mixture weights,"
                      "b = batch norm, all = entire model.)")
  return parser

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


def get_pretrain_datasets(datasets, image_size, data_dir, augment):
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


def get_combined_datasets(datasets_dict):
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


def build_models(feature_extractor, datasets, input_shape):
  """Returns combined models for the given datasets.

  Creates a tf.keras.Model that has separate inputs and top layers for all
  datasets in the experiment and shared feature extractor and I/O adapters.

  Arguments:
  feature_extractor: common feature extractor without head.
  datasets: dataset dict (from `get_datasets` method).
  input_shape: model input shape.
  """

  num_classes = [datasets[x]["num_classes"] for x in datasets]
  ds_names = list(datasets.keys())
  num_datasets = len(datasets)

  inputs = [tkf.Input(name="%s_in" % x, shape=input_shape)
            for x in ds_names]
  features = [feature_extractor(x) for x in inputs]
  outputs = [tkf.layers.Dense(num_classes[i], activation="softmax",
                              name="%s_out" % ds_names[i])(features[i])
             for i in range(num_datasets)]
  combined_model = tkf.Model(inputs, outputs, name="combined")
  return combined_model


def build_mixture_models(feature_extractor, datasets, input_shape,
                         num_layers, num_templates,
                         use_shared_mixture_weights=False,
                         share_logits=False):
  """Returns combined mixture weights models for the given datasets.

  Creates a tf.keras.Model that has separate inputs and top layers for all
  datasets in the experiment and shared feature extractor and I/O adapters.

  Arguments:
  feature_extractor: common feature extractor without head.
  datasets: dataset dict (from `get_datasets` method).
  input_shape: model input shape.
  num_layers: number of residual layers in feature extractor.
  num_templates: number of templates for mixture weights.
  use_shared_mixture_weights: if True, the mixture weights will be shared across
  domains; otherwise, mixture weights will be created for each dataset
  separately.
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
  if use_shared_mixture_weights:
    mixture_weights = get_mixture_weights("shared", inputs)
    features = [feature_extractor([x, *mixture_weights]) for x in inputs]
  else:
    mix_weights = [get_mixture_weights(name, inputs[0]) for name in ds_names]
    features = [feature_extractor([inputs[i], *mix_weights[i]])
                for i in range(num_datasets)]
  if share_logits:
    out_layer = get_output_layer(num_classes[0], "shared")
    outputs = [rename(out_layer(features[i]), "%s_out" % ds_names[i])
               for i in range(num_datasets)]
  else:
    out_layers = [get_output_layer(num_classes[i], ds_names[i])
                  for i in range(num_datasets)]
    outputs = [out_layers[i](features[i]) for i in range(num_datasets)]

  combined_model = tkf.Model(inputs, outputs, name="combined")
  return combined_model

def get_combined_model(
    datasets_info, input_shape, shared=False, share_mixture=False,
    share_logits=False, num_layers=16, num_templates=4,
    **kwargs):
    # tensor_size=16, in_adapter="strided", out_adapter="isometric",
    # dropout=0, kernel_reg=0):
  """Returns the multitask training model.

  Arguments:
  datasets_info: dataset info dictionary.
  input_shape: shape of the input image.
  shared: if True, the shared model will be created.
  share_mixture: if True, the mixture weights will be shared across
  domains.
  share_logits: if True, the logits layer will be shared.
  num_layers: number of residual blocks in the feature extractor.
  num_templates: number of templates.

  Feature extractor arguments such as:
  tensor_size: size of the resblock input tensor.
  in_adapter: input adapter type.
  out_adapter: output adapter type.
  dropout: dropout.
  kernel_regularizer: kernel regularization parameter.
  """
  if not shared:
    feature_extractor = cresnet.resnet(
        input_shape=input_shape, num_layers=num_layers,
        num_classes=10, name="feature_extractor", with_head=False,
        out_filters=[256, 512],
        **kwargs)
    combined_model = build_models(
        feature_extractor=feature_extractor, datasets=datasets_info,
        input_shape=input_shape)
  else:
    feature_extractor = sresnet.shared_resnet(
        input_shape=input_shape, num_layers=num_layers,
        num_templates=num_templates, num_classes=10, 
        with_head=False, mixture_weights_as_input=True,
        name="feature_extractor", out_filters=[256, 512], **kwargs)
    combined_model = build_mixture_models(
        feature_extractor=feature_extractor, datasets=datasets_info,
        input_shape=input_shape, num_layers=num_layers,
        num_templates=num_templates, 
        use_shared_mixture_weights=share_mixture, share_logits=share_logits)
  return combined_model


def pretrain(new_shape, data_dir, save_path, shared=False, share_mixture=False,
             share_logits=False,
             dataset_type="digits", aug=False, batch_size=32, lr=2*1e-3,
             num_epochs=200, num_steps=1500, num_datasets=1,
             tensor_size=16, num_layers=16, num_templates=4,
             in_adapter="strided", out_adapter="isometric",
             dropout=0, kernel_reg=0, lsmooth=0, restore_checkpoint=False,
             ckpt_path=None):
  """Returns a pretrained model with given parameters.

  Arguments:
  new_shape: the images will be reshaped to this size.
  data_dir: the datasets will be stored in this folder.
  save_path: the checkponts, logs and results will be saved to this folder.
  shared: If True, the shared model will be created.
  share_mixture: If True, the mixture weights are shared across tasks.
  share_logits: If True, the last logits layer will be shared across datasets.
  dataset_type: type of dataset: `digits`, `characters`, `small_natural` or
    `natural`.
  aug: whether the data augmentation should be used.
  batch_size: batch size.
  lr: initial learning rate.
  num_epochs: number of epochs.
  num_steps: number of steps per epoch.
  num_datasets: how many datasets the model should be pretrained on.
  tensor_size: size of resblock tensor.
  num_layers: number of layers in the network.
  num_templates: number of templates.
  in_adapter: input adapter type.
  out_adapter: output adapter type.
  dropout: dropout (drop).
  kernel_reg: kernel regularizer parameter.
  lsmooth: label smoothing coefficient.
  restore_checkpoint: if True, the model will be restored from the latest
    checkpoint.
  """
  h, w, _ = new_shape
  ds_list = DATASET_TYPE_DICT[dataset_type]
  if ds_list is None:
    raise ValueError("Given dataset type is not supported")

  # acquiring the combined datasets
  dataset_dict, datasets_info = get_pretrain_datasets(
      datasets=ds_list, image_size=[h, w], data_dir=data_dir, augment=aug)
  ds_list = list(dataset_dict.keys())
  print("Training the model on datasets: ", ds_list)
  comb_dataset_train, comb_dataset_test = get_combined_datasets(dataset_dict)
  comb_dataset_train = (comb_dataset_train.shuffle(buffer_size=100)
                        .batch(batch_size, drop_remainder=True))
  comb_dataset_test = (comb_dataset_test.shuffle(buffer_size=100)
                       .batch(batch_size, drop_remainder=True))
  # creating the multitask model
  combined_model = get_combined_model(
      datasets_info, new_shape, num_layers=num_layers,
      num_templates=num_templates, tensor_size=tensor_size,
      in_adapter=in_adapter, out_adapter=out_adapter,
      dropout=dropout, kernel_regularizer=kernel_reg,
      shared=shared, share_mixture=share_mixture, share_logits=share_logits)

  lr_schedule = training.get_lr_schedule(lr, num_epochs)
  losses, loss_weights = get_losses(ds_list, num_datasets, lsmooth)
  print(loss_weights)
  callbacks, ckpt = training.get_callbacks(save_path, lr_schedule, "pretrained")
  if ckpt_path is None:
    ckpt_path = ckpt

  if restore_checkpoint > 0:
    try:
      combined_model.load_weights(ckpt_path)
      print("Restored weights from %s" % ckpt_path)
    except:
      print("could not restore weights from %s" % ckpt_path)
      pass
  combined_model.summary()
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0))
  # fitting the model
  trained_model = train_model(combined_model, comb_dataset_train,
                              comb_dataset_test, callbacks=callbacks,
                              optimizer=optimizer, losses=losses,
                              loss_weights=loss_weights,
                              num_epochs=num_epochs, num_steps=num_steps,
                              batch_size=batch_size)
  # saving the model
  trained_model.save_weights(ckpt)
  return trained_model, comb_dataset_train, comb_dataset_test

def train_model(model, train_data, test_data, callbacks, optimizer,
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
      model.get_layer("feature_extractor").trainable = True
      resblock = (model.get_layer("feature_extractor")
                  .get_layer("weighted_resblock"))
      resblock.trainable = True
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

  return model
