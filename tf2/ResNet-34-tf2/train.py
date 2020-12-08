import tensorflow as tf
from dataloader import read_path, load_image
from model import resnet_34
import config as cfg
import matplotlib.pyplot as plt


def train(validation=True, load_weight=False):
    model = resnet_34(training=True)
    model.summary()

    if load_weight:
        weight_file = cfg.trained_weight
        model.load_weights(weight_file)

    model.compile(optimizer=cfg.optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.ckpt_name_training,
                                                       save_weights_only=True,
                                                       verbose=cfg.verbose,
                                                       period=cfg.save_ckpt_interval)  # 2*n epoch -> save model

    filenames_train, filenames_eval = read_path()

    train_images, train_labels = load_image(filenames_train, type='train')
    if validation:
        eval_images, eval_labels = load_image(filenames_eval, type='eval')
        history = model.fit(train_images, train_labels, epochs=cfg.num_epochs, batch_size=cfg.batch_size,
                            verbose=cfg.verbose,
                            callbacks=[ckpt_callback], validation_data=(eval_images, eval_labels))
    else:
        history = model.fit(train_images, train_labels, epochs=cfg.num_epochs, batch_size=cfg.batch_size,
                            verbose=cfg.verbose,
                            callbacks=[ckpt_callback])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'g-', label='accuracy')
    plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig("train_result.jpg")


if __name__ == "__main__":
    train(validation=True, load_weight=False)
    exit(0)
