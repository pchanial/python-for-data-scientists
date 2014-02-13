"""
Sample and theoretical Maxwell distributions.

"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as mp


def velocity2speed(speed, ndims):
    """ Return the ndims-dimensional speed of the particles. """
    return np.sqrt(np.sum(speed[..., :ndims]**2, axis=-1))


def speed_distribution(speed, ndims):
    """
    Return the probability distribution function of the ndims-dimensional
    speed of the particles.
    """
    return ((np.pi / 2)**(-np.abs(ndims-2) / 2) *
            speed**(ndims - 1) *
            np.exp(-speed**2 / 2))


NPARTICULES = 1000000

velocity = np.random.standard_normal((NPARTICULES, 3))

for ndims in (1, 2, 3):
    speed = velocity2speed(velocity, ndims)
    ax = mp.subplot(1, 3, ndims)
    n, bins, patches = ax.hist(speed, bins=100, normed=True)
    ax.set_title('{}-d speed distribution'.format(ndims))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('speed')
    ax.plot(bins, speed_distribution(bins, ndims), 'r', linewidth=2)

mp.show()
