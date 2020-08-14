import os
import tensorflow as tf
from lib import multitask as mt
from utils import args_util, training

def main():
  """Performs a finetuning or domain adaptation experiment."""
  parser = mt.get_custom_parser()
  args = parser.parse_args()
  args.save_path = os.path.abspath(args.save_path)
  exp_path = os.path.join(args.save_path, args.name)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  args_util.write_config(args, exp_path)

  if len(args.ckpt_path) == 0:
    args.ckpt_path = None
  # pretrain the model
  new_shape = (args.reshape_to, args.reshape_to, 3)
  model, train_dataset, test_dataset = mt.pretrain(
      new_shape, data_dir=args.data_dir, save_path=exp_path,
      shared=args.shared > 0, share_mixture=args.share_mixture > 0,
      share_logits=args.share_logits > 0,
      dataset_type=args.dataset_type,
      aug=args.aug > 0, batch_size=args.batch_size, lr=args.lr,
      num_epochs=args.num_epochs, num_steps=args.num_steps,
      num_datasets=args.num_datasets, tensor_size=args.size,
      num_layers=args.num_blocks, num_templates=args.num_templates,
      in_adapter=args.in_adapter_type, out_adapter=args.out_adapter_type,
      filter_base=args.out_filter_base,
      dropout=args.dropout, kernel_reg=args.kernel_reg,
      restore_checkpoint=args.restore > 0, ckpt_path=args.ckpt_path,
      lsmooth=args.lsmooth, separate_bn=args.sep_bn > 0, depth=args.depth)
  # Evaluate trained model
  results_file = os.path.join(exp_path, "results.txt")

  if "office" in args.dataset_type:
    ds_list = ["amazon", "dslr", "webcam"]
  elif "domain_net_subset" in args.dataset_type:
    ds_list = ["painting", "real", "sketch"]
  elif "domain_net" in args.dataset_type:
    ds_list = ["clipart", "infograph", "painting", "real", "quickdraw", "sketch"]
  else:
    ds_list = mt.DATASET_TYPE_DICT[args.dataset_type]

  # copying the source weights to target variables
  if args.copy_weights > 0:
    model = mt.copy_weights(model, source=ds_list[0],
                            target=args.target_dataset,
                            num_layers=args.num_blocks,
                            shared=args.shared > 0)

  num_dsets = len(ds_list)
  scores = model.evaluate(test_dataset)
  fl = open(results_file, "w")
  fl.write("Pretraining results\n")
  for i in range(num_dsets):
    print("Test loss on %s: %f" % (ds_list[i], scores[i + 1]))
    print("Test accuracy on %s: %f" % (ds_list[i], scores[num_dsets + i + 1]))
    fl.write("Test loss on %s: %f\n" % (ds_list[i], scores[i + 1]))
    fl.write("Test accuracy on %s: %f\n\n" %
             (ds_list[i], scores[num_dsets + i + 1]))
  fl.close()

  # fixing model weights and training
  fixed_model = mt.fix_weights(model, target_dataset=args.target_dataset,
                               finetune_mode=args.finetune_mode,
                               shared=args.shared > 0)
  lr_schedule = training.get_lr_schedule(args.lr*0.5, args.num_epochs_finetune,
                                         end_lr_coefficient=0.2)
  losses, loss_weights = mt.get_losses(ds_list, args.num_datasets,
                                       args.lsmooth)
  for loss in loss_weights:
    loss_weights[loss] = float(("%s_out" % args.target_dataset) == loss)
  print("Loss weights:", loss_weights)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0))
  callbacks, ckpt = training.get_callbacks(exp_path, lr_schedule, "finetuned")
  vis_path = os.path.join(exp_path, "mix_vis", "finetuned")
  vis_cbk = training.VisualizeCallback(exp_path, domains=ds_list,
                              num_templates=args.num_templates,
                              num_layers=args.num_blocks,
                              frequency=5)
  callbacks.append(vis_cbk)
  fixed_model.summary()
  # finetuning the model
  model = mt.train_model(fixed_model, train_dataset, test_dataset,
                         callbacks=callbacks, optimizer=optimizer,
                         losses=losses, loss_weights=loss_weights,
                         num_epochs=args.num_epochs_finetune,
                         num_steps=args.num_steps, batch_size=args.batch_size)

  # evaluating the model
  scores = fixed_model.evaluate(test_dataset)
  fixed_model.save_weights(ckpt)
  fl = open(results_file, "a")
  fl.write("Finetuning results\n")
  for i in range(num_dsets):
    print("Test loss on %s: %f" % (ds_list[i], scores[i + 1]))
    print("Test accuracy on %s: %f" % (ds_list[i], scores[num_dsets + i + 1]))
    fl.write("Test loss on %s: %f\n" % (ds_list[i], scores[i + 1]))
    fl.write("Test accuracy on %s: %f\n\n" %
             (ds_list[i], scores[num_dsets + i + 1]))
  fl.close()


if __name__ == "__main__":
  main()
