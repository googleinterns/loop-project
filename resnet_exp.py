""" Experiment with custom ResNet on CIFAR10 dataset."""

import os
import tensorflow as tf
from lib import custom_resnet as cresnet
from lib import shared_resnet as sresnet
from utils import args_util, training, datasets_util
from lib.resnet_parameters import ResNetParameters


def main():
  args, exp_path = args_util.get_args()
  # getting the dataset
  h = args.reshape_to
  dataset, info = datasets_util.get_dataset(
      args.target_dataset, [h, h], data_dir=args.data_dir,
      augment=args.aug > 0)
  num_classes = info["num_classes"]
  train_dataset = tf.data.Dataset.zip((dataset["train_x"],
                                       dataset["train_y"]))
  test_dataset = tf.data.Dataset.zip((dataset["test_x"],
                                      dataset["test_y"]))
  train_dataset = (train_dataset.shuffle(buffer_size=100)
                   .batch(args.batch_size, drop_remainder=True))
  test_dataset = (test_dataset.shuffle(buffer_size=100)
                  .batch(args.batch_size, drop_remainder=True))
  model_params = ResNetParameters()
  model_params.init_from_args(args)
  model_params.num_classes = num_classes
  # creating the model
  get_model_fn = cresnet.resnet if args.shared == 0 else sresnet.shared_resnet
  model = get_model_fn(model_params)

  if args.restore > 0:
    training.restore_model(args.ckpt_path, model)

  lr_schedule = training.get_lr_schedule(args.lr, args.num_epochs)
  cce_loss = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=args.lsmooth)
  model.compile(
      loss=cce_loss,
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0)),
      metrics=["acc"])
  print(model.summary())

  callbacks, ckpt = training.get_callbacks(exp_path, lr_schedule)
  # Fitting the model
  model.fit(train_dataset, batch_size=args.batch_size,
            epochs=args.num_epochs, steps_per_epoch=args.num_steps,
            validation_data=test_dataset, shuffle=True, callbacks=callbacks)

  # evaluating the model
  scores = model.evaluate(test_dataset)
  model.save_weights(ckpt)
  results_file = os.path.join(exp_path, "results.txt")
  with open(results_file, "w") as fl:
    fl.write("Pretraining results\n")
    print("Test loss: %f" % scores[0])
    print("Test accuracy: %f" % scores[1])
    fl.write("Test loss: %f\n" % scores[0])
    fl.write("Test accuracy: %f\n" % scores[1])


if __name__ == "__main__":
  main()
