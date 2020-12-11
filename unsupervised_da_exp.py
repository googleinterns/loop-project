""" This code runs unsupervised domain adaptation experiments.
"""
import os
from dataclasses import dataclass
import tensorflow as tf
import tensorflow.keras as tfk
from lib import multitask as mt
from lib import adversarial_da as adv
from lib import moment_matching_da as mm
from lib import swasserstein_da as swd
from lib.resnet_parameters import ResNetParameters
from utils import training
from utils import args_util

def get_custom_parser():
  """Returns custom argument parser.
  """
  parser = mt.get_custom_parser()
  parser.add_argument("--classifier_ckpt",
                      help="pretrained classifier checkpoint")
  parser.add_argument("--da_mode",
                      choices=["discr", "mm", "mmv2", "swd"],
                      default="discr",
                      help="domain adaptation type"
                      "(discr = discriminative"
                      "mm = moment matching"
                      "mmv2 = moment matching with discrepancy"
                      "swd = sliced Wasserstein distance)")
  parser.add_argument("--da_loss_weight",
                      type=float,
                      default=1.,
                      help="DA loss weight.")
  parser.add_argument("--copy_from",
                      type=str,
                      default="",
                      help="Target MW will be copied from this source.")
  parser.add_argument("--use_in_adapter",
                      type=int,
                      default=0,
                      help="if > 0, input adapter will be finetuned for DA")
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
  use_in_adapter: whether to finetune inpout adapter.
  """
  target_domain: str = None
  da_mode: str = "discr"
  da_loss_weight: float = 1.
  copy_from: str = ""
  block_shape: tuple = (16, 16, 4)
  num_classes: int = 10
  num_layers: int = 16
  lr: float = 1e-3
  num_epochs_da: int = 1
  use_in_adapter: int = 0
  classifier_ckpt: str = "./checkpoint.ckpt"
  share_mw_after: int = -1
  num_source_domains: int = 1

  def init_from_args(self, args):
    """Initialize fields from arguments."""
    self.target_domain = args.target_dataset
    self.num_epochs_da = args.num_epochs_finetune
    self.da_mode = args.da_mode
    self.da_loss_weight = args.da_loss_weight
    self.num_layers = args.num_blocks
    self.lr = args.lr
    self.block_shape = (args.size, args.size, args.depth)
    self.copy_from = args.copy_from if len(args.copy_from) > 0 else None
    self.use_in_adapter = args.use_in_adapter > 0
    self.classifier_ckpt = args.classifier_ckpt
    self.share_mw_after = args.share_mw_after
    self.num_source_domains = args.num_datasets


def eval_source_only(model, test_ds, datasets, target_domain, copy_from,
		     num_layers, results_file, da_mode, num_steps=1000):
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

  # copying the source weights to target variables
  _ = mt.copy_weights(so_model, source=copy_from,
                      target=target_domain,
                      num_layers=num_layers,
                      shared=True)
  scores = model.evaluate(test_ds, steps=num_steps)
  mt.write_scores(datasets, scores, results_file, f_mode="a+",
                  scores_name="Source only %s" % copy_from)

def compile_da_model(da_model, losses, loss_weights,
                     da_params=DomainAdaptationParameters(),
                     lr_schedule=1e-4):
  cl_optimizer = tfk.optimizers.RMSprop(learning_rate=lr_schedule)
  da_optimizer = tfk.optimizers.Adam(learning_rate=lr_schedule)

  if "discr" in da_params.da_mode:
    adv_optimizer = tfk.optimizers.Adam(learning_rate=lr_schedule)
    da_model.compile(
        cl_optimizer=cl_optimizer,
        d_optimizer=da_optimizer, adv_optimizer=adv_optimizer,
        loss_fn=tfk.losses.BinaryCrossentropy(from_logits=False),
        cl_losses=losses, cl_loss_weights=loss_weights, d_loss_weight=1.,
        adv_loss_weight=da_params.da_loss_weight)
  elif "swd" in da_params.da_mode:
    swd_optimizer = tfk.optimizers.Adam(learning_rate=lr_schedule)
    da_model.compile(
        swd_optimizer=swd_optimizer, swd_loss_weight=da_params.da_loss_weight,
        cl_optimizer=cl_optimizer, cl_losses=losses,
	cl_loss_weights=loss_weights)
  elif "mmv2" in da_params.da_mode:
    cl_2_optimizer = tfk.optimizers.RMSprop(learning_rate=lr_schedule)
    da_model.compile(
        cl1_optimizer=cl_optimizer, cl2_optimizer=cl_2_optimizer,
        mm_optimizer=da_optimizer, mm_loss_weight=da_params.da_loss_weight,
        cl_losses=losses, cl_loss_weights=loss_weights)
  elif "mm" in da_params.da_mode:
    da_model.compile(
        cl_optimizer=cl_optimizer, mm_optimizer=da_optimizer,
        mm_loss_weight=da_params.da_loss_weight, cl_losses=losses,
        cl_loss_weights=loss_weights)
  else:
    raise ValueError("Wrong DA model type.")
  return da_model


def get_da_model(classifier, ds_list, losses, loss_weights,
                 da_params=DomainAdaptationParameters(),
                 restore=False, ckpt_path=None,
                 model_params=ResNetParameters(),
                 lr_schedule=1e-4):
  """Returns a compiled DA model of a given type.

  Arguments:
  classifier: source classifier model.
  ds_list: list of dataset names.
  losses: dictionary of classification losses for all domains.
  loss_weights: dictionary of classification loss weights.
  lr_schedule: learning rate schedule function.
  da_params: domain adaptation parameters.
  """

  if not "mmv2" in da_params.da_mode:
    training.restore_model(da_params.classifier_ckpt, classifier)

  # creating a DA model
  if "discr" in da_params.da_mode:
    discriminator = adv.get_discriminator(
        input_shape=da_params.block_shape, num_layers=2,
        dropout=0.3, activation=tf.nn.leaky_relu)
    da_model = adv.AdversarialClassifier(
        discriminator=discriminator, classifier=classifier,
        domains=ds_list, target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from,
        finetune_in_adapter=da_params.use_in_adapter,
        shared_mw_after=da_params.share_mw_after)
  elif "swd" in da_params.da_mode:
    da_model = swd.SWDClassifier(
	classifier=classifier, domains=ds_list,
        target_domain=da_params.target_domain, num_layers=da_params.num_layers,
        copy_mix_from=da_params.copy_from, num_projections=10000,
        finetune_in_adapter=da_params.use_in_adapter,
        shared_mw_after=da_params.share_mw_after)

  elif "mmv2" in da_params.da_mode:
    ds_info = {x: {"num_classes": da_params.num_classes} for x in ds_list}
    head_weights = restore_mmv2_classifier(classifier, ds_info, model_params,
                                           da_params.classifier_ckpt)
    da_model = mm.MomentMatchingClassifierV2(
        classifier=classifier, domains=ds_list,
        target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from,
        num_classes=da_params.num_classes,
        finetune_in_adapter=da_params.use_in_adapter,
        shared_mw_after=da_params.share_mw_after)
    if not head_weights is None:
      da_model.classifier_1.get_layer("head_1").set_weights(head_weights)
      head_weights_2 = [x + tf.random.normal(x.shape, 0, 0.05)
		        for x in head_weights]
      da_model.classifier_2.get_layer("head_2").set_weights(head_weights_2)

  elif "mm" in da_params.da_mode:
    da_model = mm.MomentMatchingClassifier(
        classifier=classifier, domains=ds_list,
        target_domain=da_params.target_domain,
        num_layers=da_params.num_layers, copy_mix_from=da_params.copy_from,
        finetune_in_adapter=da_params.use_in_adapter,
        shared_mw_after=da_params.share_mw_after)
  else:
    raise ValueError("Given DA mode is not implemented.")

  da_model = compile_da_model(
      da_model, losses, loss_weights, da_params, lr_schedule)

  if restore and not ckpt_path is None:
    training.restore_model(ckpt_path, da_model)
  return da_model



def train_da_model(classifier, datasets, train_ds, test_ds, num_tr_datasets,
                   results_file, da_params=DomainAdaptationParameters(),
                   train_params=training.TrainingParameters(),
                   model_params=ResNetParameters()):
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
      datasets=datasets, num_tr_datasets=num_tr_datasets, prefix="DA",
      add_mw_callback=False, end_lr_coefficient=1., train_params=train_params)
  callbacks, lr_schedule, losses, loss_weights, ckpt = fitting_info
  # creating a DA model

  da_model = get_da_model(classifier, datasets, losses=losses,
                          loss_weights=loss_weights,
                          lr_schedule=lr_schedule,
                          da_params=da_params,
                          restore=train_params.restore > 0, ckpt_path=ckpt,
                          model_params=model_params)
  # evaluate the source-only accuracy
  eval_source_only(
      da_model, datasets=datasets, target_domain=da_params.target_domain,
      num_layers=train_params.num_layers, results_file=results_file,
      da_mode=da_params.da_mode, test_ds=train_ds,
      copy_from=da_params.copy_from, num_steps=train_params.num_steps)

  print("Training DA model on source and target")
  # fit the DA model
  da_model.train_classifier = True
  da_model.fit(
      train_ds.prefetch(tf.data.experimental.AUTOTUNE),
      epochs=da_params.num_epochs_da, steps_per_epoch=train_params.num_steps,
      validation_steps=train_params.num_steps,
      validation_data=train_ds, callbacks=callbacks)
  da_model.save_weights(ckpt)
  scores = da_model.evaluate(train_ds, steps=train_params.num_steps)
  mt.write_scores(datasets, scores, results_file, f_mode="a+",
                  scores_name="DA train")
  scores = da_model.evaluate(test_ds)
  mt.write_scores(datasets, scores, results_file, f_mode="a+",
                  scores_name="DA test")
  return da_model


def restore_mmv2_classifier(classifier, ds_info, model_params, ckpt_path):
  # restore the Moment Matching classifier from the checkpoint
  ckpt_classifier = mt.get_combined_model(
      datasets_info=ds_info, model_params=model_params, shared=True,
      share_logits=True, with_head=True, multitask=True)
  if training.restore_model(ckpt_path, ckpt_classifier):
    ft_extractor = ckpt_classifier.get_layer("feature_extractor")
    ft_extr_weights = ft_extractor.get_weights()
    classifier.get_layer("feature_extractor").set_weights(ft_extr_weights)
    for layer in ckpt_classifier.layers:
      if "_mix_" in layer.name:
        mw = layer.get_weights()
        classifier.get_layer(layer.name).set_weights(mw)
        print("Restored weights for layer %s" % layer.name)
  # return logits layer parameters
    return ckpt_classifier.get_layer("shared_out").get_weights()
  return None



def main():
  args, exp_path = args_util.get_args(get_custom_parser)
  results_file = os.path.join(exp_path, "results.txt")
  # obtaining the datasets
  ds_list = mt.DATASET_TYPE_DICT[args.dataset_type]
  h = args.reshape_to
  ds_train, ds_test, info = mt.get_datasets(
      ds_list, new_shape=[h, h], data_dir=args.data_dir, augment=args.aug > 0,
      batch_size=args.batch_size)
  ds_list = list(info.keys())
  num_classes = info[ds_list[0]]["num_classes"]
  if len(args.copy_from) == 0:
    args.copy_from = ds_list[0]
  # create the main classifier
  model_params = ResNetParameters()
  model_params.init_from_args(args)
  model_params.with_head = False
  model_params.shared = True
  model_params.num_classes = num_classes
  sharing_mw_index = None if args.share_mw_after < 0 else args.share_mw_after
  # creating the classifier model
  classifier = mt.get_combined_model(
      datasets_info=info, model_params=model_params, shared=True,
      share_logits=True, with_head=not "mmv2" in args.da_mode,
      share_mw_after=sharing_mw_index, multitask=args.multitask_resnet > 0)

  train_params = training.TrainingParameters()
  train_params.init_from_args(args)
  da_params = DomainAdaptationParameters()
  da_params.init_from_args(args)
  da_params.num_classes = num_classes
  #training the model
  train_da_model(
      classifier, datasets=ds_list, train_ds=ds_train, test_ds=ds_test,
      num_tr_datasets=args.num_datasets, results_file=results_file,
      da_params=da_params, train_params=train_params, model_params=model_params)


if __name__ == "__main__":
  main()
