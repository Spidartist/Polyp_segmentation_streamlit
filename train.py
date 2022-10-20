import os
import numpy as np
from glob import glob
import tensorflow as tf
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from keras.metrics import Accuracy, Recall, Precision
from keras.optimizers import adam_v2
from model import build_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":

    ## Seeding
    np.random.seed(42)
    tf.random.set_seed(42)


    path = "Kvasir_SEG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    # print(len(train_x), len(valid_x), len(test_x))

    # hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 10

    train_dataset = tf_dataset(train_x, train_y, batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch)

    model = build_model()
    metrics = ["acc", Recall(), Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    callbacks = [
        ModelCheckpoint("files/model_new.h5"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch
    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle=False
    )
