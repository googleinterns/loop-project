import os
from lib import multitask as mt
from utils import args_util, training
from lib.resnet_parameters import ResNetParameters


def main():
  """Performs a finetuning or supervised domain adaptation experiment."""
  args, exp_path = args_util.get_args(mt.get_custom_parser)
  results_file = os.path.join(exp_path, "results.txt")
  # Get the dataset
  ds_list = mt.DATASET_TYPE_DICT[args.dataset_type]
  img_size = args.reshape_to
  if ds_list is None:
    raise ValueError("Given dataset type is not supported")
  ds_train, ds_test, ds_info = mt.get_datasets(
      ds_list, new_shape=[img_size, img_size], data_dir=args.data_dir,
      augment=args.aug > 0, batch_size=args.batch_size)
  ds_list = list(ds_info.keys())

  # create the model
  model_params = ResNetParameters()
  model_params.init_from_args(args)
  model_params.with_head = False
  combined_model = mt.get_combined_model(
      datasets_info=ds_info, model_params=model_params, with_head=True,
      shared=args.shared > 0, share_logits=args.share_logits > 0)
  combined_model.summary()

  # pretrain the model
  train_params = training.TrainingParameters()
  train_params.init_from_args(args)
  combined_model = mt.train_model(
      combined_model, train_data=ds_train, test_data=ds_test, datasets=ds_list,
      finetune_mode=None, prefix="Pretrained", target_dataset=None,
      num_train_datasets=args.num_datasets, train_params=train_params,
      shared=args.shared > 0)

  # copying the source weights to target variables
  if args.copy_weights > 0:
    combined_model = mt.copy_weights(
        combined_model, source=ds_list[0], target=args.target_dataset,
        num_layers=args.num_blocks, shared=args.shared > 0)
  # Evaluate trained model
  scores = combined_model.evaluate(ds_test)
  mt.write_scores(ds_list, scores, results_file, f_mode="w",
                  scores_name="Pretraining")
  # finetuning the model
  if args.num_epochs_finetune > 0:
    train_params = training.TrainingParameters()
    train_params.init_from_args(args)
    train_params.lr *= 0.5
    train_params.num_epochs = args.num_epochs_finetune
    finetuned_model = mt.train_model(
        combined_model, train_data=ds_train, test_data=ds_test,
        datasets=ds_list, finetune_mode=args.finetune_mode, prefix="Finetuned",
        target_dataset=args.target_dataset, train_params=train_params,
        num_train_datasets=args.num_datasets, shared=args.shared > 0)
    # evaluating the finetuned model
    scores = finetuned_model.evaluate(ds_test)
    mt.write_scores(ds_list, scores, results_file, f_mode="a+",
                    scores_name="Finetuning")


if __name__ == "__main__":
  main()
