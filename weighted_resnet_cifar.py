import tensorflow as tf
from datetime import datetime as dt
import os
from lib import io_adapters, weighted_resblock
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=8,
                    help="tensor size along X and Y axis")
parser.add_argument("--aug_mode", type=int, choices=[0, 1, 2, 3], default=1,
                    help="augmentation mode (int from 0 to 3)")
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size (default is 128)")
parser.add_argument("--num_blocks", type=int, default=16,
                    help="number of residual blocks")
parser.add_argument("--num_templates", type=int, default=4,
                    help="number of templates (default is 4)")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout (default is 0.2)")
parser.add_argument("--out_adapter_type",
                    choices=["original", "v1", "v2", "isometric", "extended", "shallow"],
                    default="original",
                    help="Output adapter type (default is original)")
parser.add_argument("--conv_base",
                    type=int,
                    default=32,
                    help="Base number of filters in output adapters convs (default is 32)")
parser.add_argument("--label_smoothing",
                    type=float,
                    default=0.1,
                    help="Label smoothing parameter (default is 0.1)")
parser.add_argument("--kernel_reg",
                    type=float,
                    default=1e-5,
                    help="kernel regularization parameter (default is 1e-5)")
parser.add_argument("--lr",
                    type=float,
                    default=2*1e-3,
                    help="learning rate (default is 0.02)")
args = parser.parse_args()


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 2*1e-3 - epoch*1e-7
    print('Learning rate: ', lr)
    return lr

def normalize(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y

def augment1(x, y):
    x = tf.image.resize_with_crop_or_pad(x, h + 8, w + 8)
    x = tf.image.random_crop(x, [BATCH_SIZE, h, w, c])
    return x, y

def augment2(x, y):
    x = tf.image.resize_with_crop_or_pad(x, h + 8, w + 8)
    x = tf.image.random_crop(x, [BATCH_SIZE, h, w, c])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

def augment3(x, y):
    x = tf.image.resize_with_crop_or_pad(x, h + 8, w + 8)
    x = tf.image.random_crop(x, [BATCH_SIZE, h, w, c])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

h, w, c = (32, 32, 3)
BATCH_SIZE = args.batch_size
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

aug_fn = [None, augment1, augment2, augment3][args.aug_mode]

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                  .batch(BATCH_SIZE, drop_remainder=True)
                  .shuffle(buffer_size=50000)).repeat()
test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                  .batch(BATCH_SIZE, drop_remainder=True)).repeat()

test_dataset = test_dataset.map(normalize)
if aug_fn is None:
    train_dataset = train_dataset.map(normalize)

else: 
    train_dataset = train_dataset.map(aug_fn).map(normalize)

in_adapter = io_adapters.create_input_adapter((h, w, c),
                                              size=args.size, depth=40,
                                              activation=tf.nn.swish)

templates = weighted_resblock.ResBlockTemplate()

inputs = tf.keras.Input(shape=(h, w, c))
x = in_adapter(inputs)
w_res_block = weighted_resblock.WeightedResBlock(
    kernel_size=3, expansion_factor=6,
    activation='swish', num_templates=args.num_templates)
#w_res_block.trainable = False
xi_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0)

for i in range(args.num_blocks):
  # mixture weights for this layer
  xi = weighted_resblock.MixtureWeight(num_templates=args.num_templates,
                                       initializer=xi_initializer)(x)
  x = w_res_block([x, xi])

if args.out_adapter_type == "original":
    x = tf.keras.layers.Conv2D(args.conv_base, 3, activation='swish')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    
elif args.out_adapter_type == "v1":
    x = tf.keras.layers.AveragePooling2D(pool_size=args.size)(x)
    x = tf.keras.layers.Flatten()(x)
        
elif args.out_adapter_type == "v2":
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=args.size)(x)
    x = tf.keras.layers.layers.Dropout(args.dropout)(x)
    x = tf.keras.layers.Flatten()(x)
        
elif args.out_adapter_type == "isometric":
    x = tf.keras.layers.Conv2D(args.conv_base, 1, activation='swish', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=args.size)(x)
    x = tf.keras.layers.Conv2D(2*args.conv_base, 1, activation='swish', padding='valid')(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    x = tf.keras.layers.Flatten()(x)

elif args.out_adapter_type == "extended":
    x = tf.keras.layers.Conv2D(args.conv_base, 3, activation='swish', padding='same')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=args.size)(x)
    x = tf.keras.layers.Conv2D(2*args.conv_base, 1, activation='swish', padding='valid')(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    x = tf.keras.layers.Flatten()(x)
elif args.out_adapter_type == "shallow":
    x = out_adapter(x)
elif args.out_adapter_type == "space2depth":
    out_adapter = io_adapters.create_output_adapter((args.size, args.size, 40),
                                                block_size=args.size//8,
                                                pool_stride=None,
                                                activation=tf.nn.swish,
                                                depthwise=True)
    x = out_adapter(x)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

str_now = dt.now().strftime("%d_%m_%H_%M")
exp_description = '_'.join(["shared", str(args.num_blocks) + "-layers",
                          str(args.num_templates) + "-templates",
                          str(args.size) + "-size", 
                          str(args.aug_mode) + "-aug",
                          str(args.dropout) + "-do",
                          str(args.out_adapter_type) + "-adapt",
                          str(args.conv_base) + "-base",
                          str(args.label_smoothing) + "-lsmooth",
                          str(args.kernel_reg) + "-kreg",
                          str(args.lr) + "-lr",
                          str_now])
log_path = os.path.join("logs", exp_description)
checkpoint_path = os.path.join("checkpoints", exp_description + '.ckpt')

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
tboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None,
)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=50000)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0)),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
              metrics=['acc'])
history = model.fit(train_dataset, epochs=600, initial_epoch=0,
                    steps_per_epoch=200,
                    validation_data=test_dataset,
                    validation_steps=3,
                    callbacks=[lr_scheduler, tboard, cp_callback])
