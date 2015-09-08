"""
Array creation

"""
import numpy as np

N = 10
i, j = np.mgrid[0:N, 0:N]

a1 = i[i + j < N]
a2 = j[i + j < N]
