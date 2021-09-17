import numpy as np


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
