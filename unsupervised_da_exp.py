""" This code runs unsupervised domain adaptation experiments.
"""
import os
from dataclasses import dataclass
import tensorflow as tf
import tensorflow.keras as tfk
from lib import multitask as mt
from lib import adversarial_da as adv
from lib import moment_matching_da as mm
from lib.resnet_parameters import ResNetParameters
from utils import training
from utils import args_util

def get_custom_parser():
  """Returns custom argument parser.
  """
  parser = mt.get_custom_parser()
  parser.add_argument("--da_mode",
                      choices=["discr", "mm", "mmv2"],
                      default="discr",
                      help="domain adaptation type"
                      "(discr = discriminative"
                      "mm = moment matching"
                      "mmv2 = moment matching with discrepancy")
  parser.add_argument("--da_loss_weight",
                      type=float,
                      default=1.,
                      help="DA loss weight.")
  parser.add_argument("--copy_from",
                      type=str,
                      default="",
                      help="Target MW will be copied from this source.")
  return parser

@dataclass
class DomainAdaptationParameters:
  """Stores the domain adaptation parameters.

  Arguments:
  target_domain: target domain name.
  da_mode: domain adaptation mode.
  da_loss_weight: DA loss weight.
  copy_from: name of the source domain from which to copy mixture weights.
  block_shape: resblock input shape.
  num_classes: number of classes.
  """
  target_domain: str = None
  da_mode: str = "discr"
  da_loss_weight: float = 1.
  copy_from: str = ""
  block_shape: tuple = (16, 16, 4)
  num_classes: int = 10
  num_layers: int = 16
  lr: float = 2*1e-3

  def init_from_args(self, args):
    """Initialize fields from arguments."""
    self.target_domain = args.target_dataset
    self.da_mode = args.da_mode
    self.da_loss_weight = args.da_loss_weight
    self.num_layers = args.num_blocks
    self.lr = args.lr
    self.block_shape = (args.size, args.size, args.depth)
    self.copy_from = args.copy_from


def eval_source_only(model, test_ds, datasets, target_domain, num_layers,
                     results_file, da_mode):
  """Saves source-only accuracy on target for all sources.

  Arguments:
  model: trained classifier model.
  test_ds: test dataset.
  datasets: list of dataset names.
  target_domain: target domain name.
  num_layers: number of residual layers.
  results_file: results file.
  da_mode: domain adaptation mode string.
  """
  if "mmv2" in da_mode:
    so_model = model.headless
  else:
    so_model = model.classifier
  for ds in datasets:
    # copying the source weights to target variables
    _ = mt.copy_weights(so_model, source=ds,
                        target=target_domain,
                        num_layers=num_layers,
                        shared=True)
    scores = model.evaluate(test_ds)
    mt.write_scores(datasets, scores, results_file, f_mode="a+",
                    scores_name="Source only %s" % ds)


def get_da_model(classifier, ds_list, losses, loss_weights, lr_schedule,
                 da_params: DomainAdaptationParameters):
  """Returns a compiled DA model of a given type.

  Arguments:
  classifier: source classifier model.
  ds_list: list of dataset names.
  losses: dictionary of classification losses for all domains.
  loss_weights: dictionary of classification loss weights.
  lr_schedule: learning rate schedule function.
  da_params: domain adaptation parameters.
  """
  cl_optimizer = tfk.optimizers.RMSprop(learning_rate=lr_schedule(0))
  da_lr_schedule = tfk.optimizers.schedules.PolynomialDecay(
      2*1e-3, decay_steps=100, end_learning_rate=1e-6, power=1.0)
  da_optimizer = tfk.optimizers.Adam(learning_rate=da_lr_schedule)
  # creating a DA model
  if "discr" in da_params.da_mode:
    discriminator = adv.get_discriminator(
        input_shape=da_params.block_shape, num_layers=2,
        dropout=0, activation=tf.nn.leaky_relu)
    da_model = adv.AdversarialClassifier(
        discriminator=discriminator, classifier=classifier,
        domains=ds_list, target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from)
    da_model.compile(
        cl_optimizer=cl_optimizer,
        d_optimizer=da_optimizer, adv_optimizer=da_optimizer,
        loss_fn=tfk.losses.BinaryCrossentropy(from_logits=False),
        cl_losses=losses, cl_loss_weights=loss_weights, d_loss_weight=1.,
        adv_loss_weight=da_params.da_loss_weight)

  elif "mmv2" in da_params.da_mode:
    cl2_optimizer = tfk.optimizers.RMSprop(learning_rate=2*1e-3)
    da_model = mm.MomentMatchingClassifierV2(
        classifier=classifier, domains=ds_list,
        target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from,
        num_classes=da_params.num_classes)
    da_model.compile(
        cl1_optimizer=cl_optimizer, cl2_optimizer=cl2_optimizer,
        mm_optimizer=da_optimizer, mm_loss_weight=da_params.da_loss_weight,
        cl_losses=losses, cl_loss_weights=loss_weights)

  elif "mm" in da_params.da_mode:
    da_model = mm.MomentMatchingClassifier(
        classifier=classifier, domains=ds_list,
        target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from)
    da_model.compile(
        cl_optimizer=cl_optimizer, mm_optimizer=da_optimizer,
        mm_loss_weight=da_params.da_loss_weight, cl_losses=losses,
        cl_loss_weights=loss_weights)
  else:
    raise ValueError("Given DA mode is not implemented.")
  return da_model

def train_da_model(classifier, datasets, train_ds, test_ds, num_tr_datasets,
                   prefix="DA", da_params=DomainAdaptationParameters(),
                   train_params=training.TrainingParameters()):
  """Trains the domain adaptation model.

  Arguments:
  classifier: classifier model.
  datasets: list of datasets names.
  train_ds: train dataset.
  test_ds: test_dataset.
  num_tr_datasets: number of source datasets.
  da_params: domain adaptation parameters.
  prefix: experiment prefix.
  train_params: training parameters (TrainingParameters object).
  """
  # Starting DA
  fitting_info = mt.get_losses_and_callbacks(
      datasets=datasets, num_tr_datasets=num_tr_datasets, prefix=prefix,
      add_mw_callback=False, end_lr_coefficient=0.3, train_params=train_params)
  callbacks, lr_schedule, losses, loss_weights, ckpt = fitting_info
  da_model = get_da_model(classifier, datasets, losses=losses,
                          lr_schedule=lr_schedule, loss_weights=loss_weights,
                          da_params=da_params)
  print("Training DA model on source and target")
  # fit the DA model
  da_model.fit(
      train_ds.prefetch(tf.data.experimental.AUTOTUNE),
      epochs=train_params.num_epochs, steps_per_epoch=train_params.num_steps,
      validation_data=test_ds, callbacks=callbacks)
  da_model.save_weights(ckpt)
  return da_model

def main():
  args, exp_path = args_util.get_args(get_custom_parser)
  results_file = os.path.join(exp_path, "results.txt")
  if len(args.copy_from) == 0:
    args.copy_from = None

  # obtaining the datasets
  ds_list = mt.DATASET_TYPE_DICT[args.dataset_type]
  h = args.reshape_to
  ds_train, ds_test, info = mt.get_datasets(
      ds_list, new_shape=[h, h], data_dir=args.data_dir, augment=args.aug > 0,
      batch_size=args.batch_size)
  ds_list = list(info.keys())
  # create the main classifier
  model_params = ResNetParameters()
  model_params.init_from_args(args)
  model_params.with_head = False
  model_params.shared = True
  model_params.activation = tf.nn.leaky_relu
  classifier = mt.get_combined_model(
      datasets_info=info, model_params=model_params, shared=args.shared,
      share_logits=True, with_head=not "mmv2" in args.da_mode)
  classifier.summary()

  if args.num_epochs > 0 and not "mmv2" in args.da_mode:
    print("Pretraining the model on source domains")
    train_params = training.TrainingParameters()
    train_params.init_from_args(args)
    classifier = mt.train_model(
        classifier, train_data=ds_train, test_data=ds_test,
        datasets=ds_list, finetune_mode=None, shared=args.shared > 0,
        prefix="Pretrained", target_dataset=None,
        num_train_datasets=args.num_datasets, train_params=train_params)
    scores = classifier.evaluate(ds_test)
    mt.write_scores(ds_list, scores, results_file, f_mode="w+",
                    scores_name="Pretraining")
  # Starting DA
  num_classes = info[ds_list[0]]["num_classes"]
  # fit the DA model
  train_params = training.TrainingParameters()
  train_params.init_from_args(args)
  train_params.num_epochs = args.num_epochs_finetune
  da_params = DomainAdaptationParameters()
  da_params.init_from_args(args)
  da_params.num_classes = num_classes
  da_model = train_da_model(
      classifier, datasets=ds_list, train_ds=ds_train, test_ds=ds_test,
      num_tr_datasets=args.num_datasets, prefix="DA",
      da_params=da_params, train_params=train_params)
  # Evaluate DA results
  scores = da_model.evaluate(ds_test)
  mt.write_scores(ds_list, scores, results_file, f_mode="a+",
                  scores_name="DA")
  # evaluate source-only results
  eval_source_only(
      da_model, datasets=ds_list, target_domain=args.target_dataset,
      num_layers=args.num_blocks, results_file=results_file,
      da_mode=args.da_mode, test_ds=ds_test)


if __name__ == "__main__":
  main()
