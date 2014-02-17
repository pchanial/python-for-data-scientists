"""
Example of indirect sort.

"""
import numpy as np

NPEOPLE = 1000000

people_dtype = [('name', 'S10'), ('age', int)]
people = np.empty(NPEOPLE, dtype=people_dtype)
people['name'] = ['id{}'.format(_) for _ in xrange(1, NPEOPLE + 1)]
people['age'] = np.random.random_integers(20, 70, NPEOPLE)

#==========
# method 1
#==========
# not specific to structured dtype
index = np.argsort(people['age'])
people_sorted = people[index]

# note that in some situations, the numpy function 'take' can shamefully be
# much faster than indexing with the brackets
# >>> %timeit np.take(people, index)
# 10 loops, best of 3: 42 ms per loop
# >>> %timeit people[index]
# 1 loops, best of 3: 265 ms per loop
#
# Use 'take' only if required in time critical code regions, since it makes
# the code less readable, and this performance issue is likely to be fixed
# in a future numpy version.

#==========
# method 2
#==========
# specific to structured dtype (but slower in numpy 1.8.0...):
people_sorted = np.sort(people, order='age')
