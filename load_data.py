import os
import numpy as np
import os

def load_data(path):
    data = []
    for file in os.listdir(path):
        temp_data = np.load(os.path.join(path, file))
        data.append(temp_data)

    data = np.array(data)
    data = data.reshape(-1,75,80)

    data_len = len(data)
    test_size = round(data_len*0.25)
    index = np.random.permutation(data_len)
    normalized_data = data/np.max(np.abs(data), axis=0)
    x_test = normalized_data[index[:test_size]]
    x_train = normalized_data[index[test_size:]]
    x_train = np.reshape(x_train, (len(x_train), 75, 80, 1))
    x_test = np.reshape(x_test, (len(x_test), 75, 80, 1))

    return x_train, x_test