""" This code runs unsupervised domain adaptation experiments.
"""
import os
import tensorflow as tf
import tensorflow.keras as tfk
from lib import multitask as mt
from lib import adversarial_da as adv
from lib import moment_matching_da as mm
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
  return parser

def get_da_model(classifier, num_layers,
                 ds_list, target_domain,
                 losses, loss_weights,
                 da_mode="discr", block_shape=(16, 16, 40),
                 da_loss_weight=1.,
                 lrate=0.002):
  """Returns a compiled DA model of a given type.

  Arguments:
  classifier: source classifier model.
  num_layers: number of residual blocks.
  ds_list: list of dataset names.
  target_domain: target domain name.
  losses: dictionary of classification losses for all domains.
  loss_weights: dictionary of classification loss weights.
  da_mode: domain adaptation mode: discr, mm or mmv2.
  block_shape: residual block output shape.
  da_loss_weight: DA loss weight.
  """
  cl_optimizer = tfk.optimizers.RMSprop(learning_rate=lrate)
# creating a DA model
  if "discr" in da_mode:
    discriminator = adv.get_discriminator(
        input_shape=block_shape, num_layers=2,
        dropout=0, activation=tf.nn.leaky_relu)

    da_model = adv.AdversarialClassifier(
        discriminator=discriminator, classifier=classifier,
        domains=ds_list, target_domain=target_domain,
        num_layers=num_layers)
    da_model.compile(
        cl_optimizer=cl_optimizer,
        d_optimizer=tfk.optimizers.SGD(learning_rate=0.0002),
        adv_optimizer=tfk.optimizers.Adam(learning_rate=0.0002),
        loss_fn=tfk.losses.BinaryCrossentropy(from_logits=False),
        cl_losses=losses,
        cl_loss_weights=loss_weights,
        d_loss_weight=1.,
        adv_loss_weight=da_loss_weight)

  elif "mmv2" in da_mode:
    raise NotImplementedError("mmv2 is not implemented yet.")
  else:
    da_model = mm.MomentMatchingClassifier(
        classifier=classifier,
        domains=ds_list, target_domain=target_domain,
        num_layers=num_layers)
    da_model.compile(
        cl_optimizer=cl_optimizer,
        mm_optimizer=tfk.optimizers.Adam(learning_rate=0.001),
        mm_loss_weight=1,
        cl_losses=losses,
        cl_loss_weights=loss_weights)
  return da_model



def main():
  parser = get_custom_parser()
  args = parser.parse_args()
  args.save_path = os.path.abspath(args.save_path)
  exp_path = os.path.join(args.save_path, args.name)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  args_util.write_config(args, exp_path)
  results_file = os.path.join(exp_path, "results.txt")

  if len(args.ckpt_path) == 0:
    args.ckpt_path = None
  # obtaining the datasets
  ds_list = mt.DATASET_TYPE_DICT[args.dataset_type]
  h = args.reshape_to
  dataset_dict, info = mt.get_pretrain_datasets(
      ds_list, image_size=[h, h],
      data_dir=args.data_dir, augment=args.aug > 0)
  ds_list = list(dataset_dict.keys())

  ds_train, ds_test = mt.get_combined_datasets(dataset_dict)
  ds_train = (ds_train.shuffle(buffer_size=100)
              .batch(args.batch_size, drop_remainder=True))
  ds_test = (ds_test.shuffle(buffer_size=100)
             .batch(args.batch_size, drop_remainder=True))
  # create the main classifier
  classifier = mt.get_combined_model(
      info, [h, h, 3], num_layers=args.num_blocks,
      num_templates=args.num_templates, tensor_size=args.size,
      in_adapter=args.in_adapter_type, out_adapter=args.out_adapter_type,
      dropout=args.dropout, kernel_regularizer=args.kernel_reg,
      shared=args.shared > 0, share_mixture=args.share_mixture > 0,
      share_logits=args.share_logits > 0, separate_bn=args.sep_bn > 0,
      out_filters=[args.out_filter_base, 2 * args.out_filter_base],
      depth=args.depth, activation=tf.nn.leaky_relu)

  lr_schedule = training.get_lr_schedule(args.lr, args.num_epochs, 0.5)
  losses, loss_weights = mt.get_losses(
      ds_list, args.num_datasets, args.lsmooth)
  print(loss_weights)
  callbacks, ckpt = training.get_callbacks(exp_path, lr_schedule, "checkpoint")

  if args.ckpt_path is None:
    args.ckpt_path = ckpt
  # create the DA model
  da_model = get_da_model(
      classifier, args.num_blocks, ds_list,
      target_domain=args.target_dataset,
      losses=losses, loss_weights=loss_weights,
      da_mode=args.da_mode,
      block_shape=(args.size, args.size, args.depth),
      da_loss_weight=args.da_loss_weight,
      lrate=args.lr)

  if args.restore > 0:
    try:
      da_model.load_weights(args.ckpt_path)
      print("Restored weights from %s" % args.ckpt_path)
    except ValueError:
      print("could not restore weights from %s" % args.ckpt_path)
      pass
  # fit the DA model
  da_model.fit(
      ds_train.prefetch(tf.data.experimental.AUTOTUNE),
      epochs=args.num_epochs,
      steps_per_epoch=args.num_steps,
      validation_data=ds_test,
      callbacks=callbacks)
  # Evaluate DA results
  scores = da_model.evaluate(ds_test)
  print(scores)
  mt.write_scores(ds_list, scores, results_file, f_mode="w",
                  scores_name="DA")

  # evaluate source-only results
  for ds in ds_list:
    # copying the source weights to target variables
    so_model = mt.copy_weights(da_model.classifier, source=ds,
                               target=args.target_dataset,
                               num_layers=args.num_blocks,
                               shared=args.shared > 0)

    scores = so_model.evaluate(ds_test)
    mt.write_scores(ds_list, scores, results_file, f_mode="w+",
                    scores_name="Source only %s" % ds)



if __name__ == "__main__":
  main()
