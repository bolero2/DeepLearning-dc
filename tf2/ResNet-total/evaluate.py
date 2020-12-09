import tensorflow as tf
from dataloader import read_path, load_image
import config as cfg


def evaluate():
    filenames_train, filenames_eval = read_path()
    eval_images, eval_labels = load_image(filenames_eval, type='eval')
    
    if cfg._model == "resnet34":
        from models/resnet34 import resnet_34
        model = resnet_34(training=False)
    elif cfg._model == "resnet101":
        from models/resnet101 import resnet_101
        model = resnet_101(training=False)
    elif cfg._model == "resnet152":
        from models/resnet152 import resnet_152
        model = resnet_152(training=False)

    weight_file = cfg.trained_weight
    model.load_weights(weight_file)
    model.compile(optimizer=cfg.optimizer,
                  loss=cfg.loss_function,
                  metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8), 'accuracy'])

    loss, recall_at_precision, acc = model.evaluate(eval_images, eval_labels, verbose=cfg.verbose)

    print("Loss: {:5.5f}".format(loss))
    print("Recall At Precision: {:5.2f}".format(recall_at_precision))
    print("Accuracy: {:5.2f}%".format(100 * acc))


if __name__ == "__main__":
    evaluate()
    exit(0)
