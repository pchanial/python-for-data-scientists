"""
Compute pi using the Madhava formula.

"""
from __future__ import division
import numpy as np

N = 30
k = np.arange(N)
pi = np.sqrt(12) * np.sum((-1/3)**k / (2 * k + 1))
print 'Relative error:', (pi - np.pi) / np.pi

# Note this quirk (cannot be fixed by importing division from __future__):
# In Python: 3**(-1) == 0.3333333333333333
# but in Numpy: 3**np.array(-1) == 0
# so to express (-3)**(-k) correctly: either -3 or k should be converted to float.
