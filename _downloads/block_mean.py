"""
Block mean computation.

"""
from __future__ import division
import numpy as np


def block_mean(vector):
    if vector.size % 100 != 0:
        raise ValueError('Invalid size.')
    nblocks = vector.size // 100
    return np.array([np.mean(vector[i*100:(i+1)*100]) for i in range(nblocks)])


def block_mean_fast(vector):
    if vector.size % 100 != 0:
        raise ValueError('Invalid size.')
    return np.mean(vector.reshape((N, 100)), axis=-1)


N = 30

vector = np.random.random_sample(100 * N)
print block_mean(vector)
print block_mean_fast(vector)
