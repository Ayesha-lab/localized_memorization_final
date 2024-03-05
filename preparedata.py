import numpy as np


def outlier(value):
    new = np.random.choice([i for i in range(10) if i != value])
    return new


def addnoise(data, percent):
    # set random seed for generating random % outliers
    size = len(data)
    no_outliers = int(size * (percent / 100))
    seed_value = 42
    np.random.seed(seed_value)
    indices = np.random.permutation(no_outliers)
    data[indices] = [outlier(value) for value in data[indices]]

    return data
