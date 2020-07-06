import argparse
from datetime import datetime as dt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from lib import custom_resnet

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=8,
                    help="tensor size along X and Y axis")
parser.add_argument("--aug", type=bool, default=1,
                    help="whether to use data augmentation (default 1)")
parser.add_argument("--pixel_mean", type=bool, default=1,
                    help="whether to subtract the pixel mean (default 1)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size (default is 32)")
parser.add_argument("--num_blocks", type=int, default=16,
                    help="number of residual blocks")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout (default is 0.2)")
parser.add_argument("--in_adapter_type",
                    choices=["original", "space2depth", "strided"],
                    default="original",
                    help="Output adapter type (default is original)")
parser.add_argument("--out_adapter_type",
                    choices=["v1", "v2", "isometric", "dethwise"],
                    default="v1",
                    help="Output adapter type (default is v1)")
parser.add_argument("--lsmooth",
                    type=float,
                    default=0,
                    help="Label smoothing parameter (default is 0)")
parser.add_argument("--kernel_reg",
                    type=float,
                    default=1e-5,
                    help="kernel regularization parameter (default is 1e-5)")
parser.add_argument("--lr",
                    type=float,
                    default=1e-3,
                    help="learning rate (default is 0.01)")
args = parser.parse_args()

h, w, c = (32, 32, 3)
num_classes = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if args.pixel_mean:
  x_train_mean = np.mean(x_train, axis=0)
  x_train -= x_train_mean
  x_test -= x_train_mean

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  def __init__(self, log_dir, **kwargs):
    super().__init__(log_dir=log_dir, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs.update({'lr': K.eval(self.model.optimizer.lr)})
    super().on_epoch_end(epoch, logs)

def lr_schedule(epoch):
  """Learning Rate Schedule
  Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
  Called automatically every epoch as part of callbacks during training.
  # Arguments
      epoch (int): The number of epochs
  # Returns
      lr (float32): learning rate"""
  lr = args.lr
  if epoch > 180:
    lr *= 0.5e-3
  elif epoch > 160:
    lr *= 1e-3
  elif epoch > 120:
    lr *= 1e-2
  elif epoch > 80:
    lr *= 1e-1
  print('Learning rate: ', lr)
  return lr

model = custom_resnet.resnet([h, w, c], num_layers=args.num_blocks, num_classes=10,
                             depth=40, in_adapter=args.in_adapter_type,
                             out_adapter=args.out_adapter_type, tensor_size=args.size,
                             kernel_regularizer=args.kernel_reg, dropout=args.dropout)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.lsmooth),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0)),
    metrics=['acc'])

print(model.summary())
str_now = dt.now().strftime("%d_%m_%H_%M")
exp_description = '_'.join(["custom", str(args.num_blocks) + "-layers",
                            str(args.size) + "-size", 
                            str(args.batch_size) + "batch",
                            str(args.aug) + "-aug",
                            str(args.pixel_mean) + "-pix",
                            str(args.dropout) + "-do",
                            str(args.in_adapter_type) + "-in_adapter",
                            str(args.out_adapter_type) + "-out_adapter",
                            str(args.lsmooth) + "-lsmooth",
                            str(args.kernel_reg) + "-kreg",
                            str(args.lr) + "-lr",
                            str_now])
log_path = os.path.join("logs", exp_description)
checkpoint_path = os.path.join("checkpoints", exp_description + '.ckpt')

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0,
    patience=5, min_lr=0.5e-6)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
tboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None,
)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=100000)
tftboard_callback = LRTensorBoard(log_dir=log_path)

callbacks = [cp_callback, lr_reducer, lr_scheduler, tftboard_callback]

# Run training, with or without data augmentation.
if not args.aug:
  print('Not using data augmentation.')
  model.fit(x_train, y_train,
            batch_size=args.batch_size,
            epochs=200,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=callbacks)
else:
  print('Using real-time data augmentation.')
  # This will do preprocessing and realtime data augmentation:
  datagen = ImageDataGenerator(
      # set input mean to 0 over the dataset
      featurewise_center=False,
      # set each sample mean to 0
      samplewise_center=False,
      # divide inputs by std of dataset
      featurewise_std_normalization=False,
      # divide each input by its std
      samplewise_std_normalization=False,
      # apply ZCA whitening
      zca_whitening=False,
      # epsilon for ZCA whitening
      zca_epsilon=1e-06,
      # randomly rotate images in the range (deg 0 to 180)
      rotation_range=0,
      # randomly shift images horizontally
      width_shift_range=0.1,
      # randomly shift images vertically
      height_shift_range=0.1,
      # set range for random shear
      shear_range=0.,
      # set range for random zoom
      zoom_range=0.,
      # set range for random channel shifts
      channel_shift_range=0.,
      # set mode for filling points outside the input boundaries
      fill_mode='nearest',
      # value used for fill_mode = "constant"
      cval=0.,
      # randomly flip images
      horizontal_flip=True,
      # randomly flip images
      vertical_flip=False,
      # set rescaling factor (applied before any other transformation)
      rescale=None,
      # set function that will be applied on each input
      preprocessing_function=None,
      # image data format, either "channels_first" or "channels_last"
      data_format=None,
      # fraction of images reserved for validation (strictly between 0 and 1)
      validation_split=0.0)

  # Compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied).
  datagen.fit(x_train)

  # Fit the model on the batches generated by datagen.flow().
  model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                      validation_data=(x_test, y_test),
                      epochs=200, verbose=1, workers=4,
                      callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
