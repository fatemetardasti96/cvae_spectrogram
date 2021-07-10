import numpy as np


def apply_reshaping(x_train, x_test, input_shape):

    x_train = np.resize(x_train, new_shape=(x_train.shape[0], input_shape[0], input_shape[1], 1))
    x_test = np.resize(x_test, new_shape=(x_test.shape[0], input_shape[0], input_shape[1], 1))

    return x_train, x_test