""" Experiment with custom ResNet on CIFAR10 dataset."""

import os
import numpy as np
import tensorflow as tf
from lib import custom_resnet, shared_resnet
from utils import args_util, training


def main():
  parser = args_util.get_parser()
  args = parser.parse_args()
  args.save_path = os.path.abspath(args.save_path)
  exp_path = os.path.join(args.save_path, args.name)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  args_util.write_config(args, exp_path)

  # getting the dataset
  h, w, c = (args.reshape_to, args.reshape_to, 3)
  dataset_info = training.get_dataset("cifar10", [h, w],
                                      data_dir=args.data_dir,
                                      augment=args.aug > 0)
  num_classes = dataset_info["num_classes"]
  train_dataset = tf.data.Dataset.zip((dataset_info["train_x"],
                                       dataset_info["train_y"]))
  test_dataset = tf.data.Dataset.zip((dataset_info["test_x"],
                                      dataset_info["test_y"]))
  train_dataset = (train_dataset.shuffle(buffer_size=20000)
                   .batch(args.batch_size, drop_remainder=True))
  test_dataset = (test_dataset.shuffle(buffer_size=20000)
                  .batch(args.batch_size, drop_remainder=True))

  # creating the model
  if args.shared == 0:
    model = custom_resnet.resnet([h, w, c], num_layers=args.num_blocks,
                                 num_classes=num_classes, depth=40,
                                 in_adapter=args.in_adapter_type,
                                 out_adapter=args.out_adapter_type,
                                 tensor_size=args.size,
                                 kernel_regularizer=args.kernel_reg,
                                 dropout=args.dropout)
  else:
    model = shared_resnet.shared_resnet(
        [h, w, c], num_layers=args.num_blocks, num_classes=10,
        num_templates=args.num_templates, depth=40,
        in_adapter=args.in_adapter_type, out_adapter=args.out_adapter_type,
        tensor_size=args.size, kernel_regularizer=args.kernel_reg,
        dropout=args.dropout, out_filters=[128, 256])
  
  lr_schedule = training.get_lr_schedule(args.lr, args.num_epochs)
  cce_loss = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=args.lsmooth)
  model.compile(
     loss=cce_loss,
     optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0)),
      metrics=['acc'])
  print(model.summary())

  callbacks, ckpt = training.get_callbacks(exp_path, lr_schedule)
  # Fitting the model
  model.fit(train_dataset, batch_size=args.batch_size,
            epochs= args.num_epochs, steps_per_epoch=args.num_steps,
            validation_data=test_dataset, shuffle=True, callbacks=callbacks)

  # evaluating the model
  scores = model.evaluate(test_dataset)
  model.save_weights(ckpt)
  results_file = os.path.join(exp_path, "results.txt")
  with open(results_file, "w") as fl:
    fl.write("Pretraining results\n")
    print('Test loss: %f' % scores[0])
    print('Test accuracy: %f' % scores[1])
    fl.write('Test loss: %f\n' % scores[0])
    fl.write('Test accuracy: %f\n' % scores[1])
    

if __name__ == "__main__":
  main()
