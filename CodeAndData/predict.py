import os

import keras.models
import numpy as np
import tensorflow as tf
import cv2
import keras
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm
from data import load_data, tf_dataset
from train import iou
import matplotlib.pyplot as plt


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x

# def read_image_2(path):
#     x = cv2.imread(path, cv2.IMREAD_COLOR)
#     x = cv2.resize(x, (256, 256))
#     print(np.max(x, axis=2))
#     x = x / 255.0
#     print(np.max(x, axis=2))
#     return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":

    # Dataset
    path = "Kvasir_SEG/"
    batch_size = 16
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x) // batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = keras.models.load_model("files/model_new.h5")

    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)

    # for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
    #     x = read_image(x)
    #     print(x.shape)
    #     print(x[1][1][0])
    #     print(x[1][1][1])
    #     print(x[1][1][2])
    #     break
    # x = cv2.imread("E:/Download_Photos/000-1549030079-1-1.jpg", cv2.IMREAD_COLOR)
    # x = read_image("E:/Download_Photos/271614017_1122620325160604_2071765014128705275_n.png")
    # print(np.max(x, axis=2))
    # y = read_image_2("E:/Download_Photos/271614017_1122620325160604_2071765014128705275_n.png")
    # print(np.max(y, axis=2))
    # cv2.imshow("Ima", x)
    # cv2.imshow("Ima2", y)
    #
    # cv2.waitKey(0)
