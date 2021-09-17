from funcs.functions import *


# First, import the training set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Number of training examples: m_train = 209
# Number of testing examples: m_test = 50
# Height/Width of each image: num_px = 64
# Each image is of size: (64, 64, 3)
# train_set_x shape: (209, 64, 64, 3)
# train_set_y shape: (1, 209)
# test_set_x shape: (50, 64, 64, 3)
# test_set_y shape: (1, 50)

# now I'm gonna "flatten" the training set
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# Caution! The flattening process cannot be X_flatten = X.reshape(-1, X.shape[0])

# now the shape of matrixes:
# train_set_x_flatten shape: (12288, 209)
# train_set_y shape: (1, 209)
# test_set_x_flatten shape: (12288, 50)
# test_set_y shape: (1, 50)

# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel
# so the pixel value is actually a vector of three numbers ranging from 0 to 255.
# now standarize the data
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255
