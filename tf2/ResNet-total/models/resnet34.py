import tensorflow as tf
import sys
sys.path.append("..")
import utils.config as cfg


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


def resnet_34(training):
    input = tf.keras.Input(shape=(cfg.image_size, cfg.image_size, cfg.channel))
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu',
                                 input_shape=(cfg.image_size, cfg.image_size, cfg.channel))(input)
    out = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(out)

    out = block_residual(out, 64, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 64, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 64, kernel_size=3, stride=1, training=training)

    out = block_residual(out, 128, kernel_size=3, stride=2, training=training)
    out = block_residual(out, 128, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 128, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 128, kernel_size=3, stride=1, training=training)

    out = block_residual(out, 256, kernel_size=3, stride=2, training=training)
    out = block_residual(out, 256, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 256, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 256, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 256, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 256, kernel_size=3, stride=1, training=training)

    out = block_residual(out, 512, kernel_size=3, stride=2, training=training)
    out = block_residual(out, 512, kernel_size=3, stride=1, training=training)
    out = block_residual(out, 512, kernel_size=3, stride=1, training=training)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(cfg.dropout_rate)(out)
    output = tf.keras.layers.Dense(units=cfg.label_size, activation='softmax')(out)

    model = tf.keras.Model(input, output)

    return model
