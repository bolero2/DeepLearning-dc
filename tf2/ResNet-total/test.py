import tensorflow as tf
import cv2
import numpy as np

from utils.dataloader import read_path, load_image
import utils.config as cfg


def test():
    if cfg._model == "resnet34":
        from models.resnet34 import resnet_34
        model = resnet_34(training=False)
    elif cfg._model == "resnet101":
        from models.resnet101 import resnet_101
        model = resnet_101(training=False)
    elif cfg._model == "resnet152":
        from models.resnet152 import resnet_152
        model = resnet_152(training=False)

    weight_file = cfg.trained_weight
    model.load_weights(weight_file)
    model.compile(optimizer=cfg.optimizer,
                  loss=cfg.loss_function,
                  metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8), 'accuracy'])

    if cfg.channel == 1:
        test_image = cv2.imread(cfg.TestImage, 0) / 255.0
    else:
        test_image = cv2.imread(cfg.TestImage, 1) / 255.0
    test_image = cv2.resize(test_image, (cfg.image_size, cfg.image_size))
    test_image = tf.reshape(test_image, (1, cfg.image_size, cfg.image_size, cfg.channel))

    pred = model.predict(test_image)

    label = np.where(pred[0] == np.max(pred[0]))[0]
    acc = round(np.max(pred[0]) * 100, 2)

    print(f"*** Test image result ***\nLabel: {label} \nAcuracy: {acc}%")

    import matplotlib.pyplot as plt

    plt_img = plt.imread(cfg.TestImage)
    plt.imshow(plt_img)
    plt.xlabel(f"Label: {label} \nAcuracy: {acc}%")

    plt.savefig("test_result.jpg")


if __name__ == "__main__":
    test()
    exit(0)
