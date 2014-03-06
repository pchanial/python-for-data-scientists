"""
Matrix examples

"""
import numpy as np

M = 5
N = 9
I = np.arange(M)
J = np.arange(N)

# method 1:
A = np.add.outer(I, J)
B = np.multiply.outer(I, J)

# method 2:
A = I[:, None] + J  # same as np.add(I[:, None], J)
B = I[:, None] * J  # same as np.multiply(I[:, None], J)
