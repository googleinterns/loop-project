import tensorflow as tf
import lib.io_adapters as ioad
import lib.res_block as block
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def resnet(input_shape, num_layers=20, num_classes=10, depth=40,
             in_adapter="original", out_adapter="v1", tensor_size=8, 
             kernel_regularizer=1e-5, dropout=0):
    """ResNet v1 with custom resblock model builder 

    Arguments
      input_shape (tensor): shape of input image tensor
      num_layers (int): number of residual blocks
      num_classes (int): number of classes
      depth: depth of resblock tensors
      in_adapter (string): input adapter architecture. The options are
        `original`, `space2depth` and `strided`.
      out_adapter (string): output adapter architecture. The options are
        `v1`, `v2`, `isometric` and `dethwise`.
      kernel_regularizer: kernel regularization parameter.
      dropout: dropout parameter (drop).
    Returns
      model (Model): Keras model instance
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')
    if in_adapter == 'original':
        x = layers.Conv2D(depth, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(1e-4)
                          )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    elif in_adapter == 'space2depth':
        in_adapter = ioad.create_input_adapter(
            input_shape, size=tensor_size, depth=depth, activation='relu')
        x = in_adapter(inputs)
    elif in_adapter == 'strided':
        k = max(input_shape[0]//tensor_size, input_shape[0]//tensor_size)
        in_adapter = ioad.create_input_adapter_strided(
            input_shape, input_shape, filters=40, kernel=k,
            strides=k, activation='relu')
        x = in_adapter(inputs)
    else:
        raise ValueError("Given input adapter type is not supported.")

    # Instantiate the stack of residual units
    for stack in range(num_layers):
       x = block.ResBlock(kernel_size=3, expansion_factor=6,
                          activation='swish',
                          kernel_reg=kernel_regularizer)(x)

    if out_adapter == 'v1':
        # v1 does not use BN after last shortcut connection-ReLU
        x = layers.AveragePooling2D(pool_size=8)(x)
        x = layers.Dropout(dropout)(x)
    elif out_adapter == 'v2':
        # v2 has BN-ReLU before Pooling
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.AveragePooling2D(pool_size=8)(x)
        x = layers.Dropout(dropout)(x)
    elif out_adapter == 'isometric':
        t_shape = input_shape[0] if in_adapter == 'original' else tensor_size
        in_shape = (tensor_size, tensor_size, depth)
        out_adapter = create_output_adapter_isometric(
            in_shape, filters=[128, 256], pool_size=8, activation='relu',
            dropout=dropout)
        x = out_adapter(x)
    elif out_adapter == 'dethwise':
        t_shape = input_shape[0] if in_adapter == 'original' else tensor_size
        in_shape = (tensor_size, tensor_size, depth)
        out_adapter = create_output_adapter(
            in_shape,  block_size=2, pool_stride=None,
            activation='relu', depthwise=True, dropout=dropout)
        x = out_adapter(x)
    else:
        raise ValueError("Given output adapter type is not supported.")
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model