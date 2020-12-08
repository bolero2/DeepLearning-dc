import tensorflow as tf
import config as cfg


def block_residual(input_data, filters, kernel_size, stride=1, training=True):
    residual = input_data

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(input_data)
    x = tf.keras.layers.BatchNormalization(trainable=training)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=training)(x)

    if input_data.shape[-1] != x.shape[-1]:
        residual = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(input_data)
    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.Activation('relu')(x)

    return x


def block_residual_proposed(input_data, filters, kernel_size, stride=1, training=True):
    residual = input_data

    x = tf.keras.layers.Conv2D(int(filters / 4), 1, strides=stride, padding='same')(input_data)
    x = tf.keras.layers.BatchNormalization(trainable=training)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(int(filters / 4), kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=training)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(trainable=training)(x)

    if input_data.shape[-1] != x.shape[-1]:
        residual = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(input_data)
    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.Activation('relu')(x)

    return x


def resnet_101(training):
    input = tf.keras.Input(shape=(cfg.image_size, cfg.image_size, cfg.channel))
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu',
                                 input_shape=(cfg.image_size, cfg.image_size, cfg.channel))(input)
    out = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(out)

    for c in range(3):
        stride = 1
        out = block_residual_proposed(out, 256, kernel_size=3, stride=stride, training=training)

    for c in range(4):
        if c == 0:
            stride = 2
        else:
            stride = 1
        out = block_residual_proposed(out, 512, kernel_size=3, stride=stride, training=training)

    for c in range(23):
        if c == 0:
            stride = 2
        else:
            stride = 1
        out = block_residual_proposed(out, 1024, kernel_size=3, stride=stride, training=training)

    for c in range(3):
        if c == 0:
            stride = 2
        else:
            stride = 1
        out = block_residual_proposed(out, 2048, kernel_size=3, stride=stride, training=training)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(cfg.dropout_rate)(out)
    output = tf.keras.layers.Dense(units=cfg.label_size, activation='softmax')(out)

    model = tf.keras.Model(input, output)

    return model