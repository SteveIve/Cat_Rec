import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)  # derive this by yourself
    return ds


def image2vector(image):
    # the shape of a "image" is (length, height, depth)
    length = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]

    vector = image.reshape(length * height * depth, 1)  # column vector
    return vector


def normalizeRows(x):
    # x is a numpy matrix of shape (n, m)
    norm = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    # axis=1 means calculate by rows
    # ord=2 means the norm num is 2
    x = x / norm
    return x


def softmax(x):
    x = np.exp(x)
    x_sum = np.sum(x, axis=1, keepdims=True)    # axis=1, by rows
    s = x / x_sum

    # return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    return s


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes