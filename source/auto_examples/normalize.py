"""
Normalization by the euclidian norm.

"""
import numpy as np

def normalize(v):
    return v / np.sqrt(np.sum(v**2, axis=-1))[..., None]
