"""
non-linear fit example with 3 parameters model 1/f noise :
   variance*[ 1 + (f_knee/f)^alpha ]

Jm. Colley
"""
from __future__ import division, print_function
import numpy as np
import scipy.optimize as spo
from matplotlib import pyplot as mp
from astropy.io import fits

FREQ_SAMPLING = 180


def spectrum_estimate(signal):
    """
    Return power spectrum mode 0 to Nyquist
    """
    size_signal = signal.size
    fft_signal = np.fft.fft(signal)
    ps = abs(fft_signal)**2 / size_signal
    return ps[:1 + size_signal // 2]

# inspect the FITS Table
fits.info('data_oof.fits')
hdulist = fits.open('data_oof.fits')
print(hdulist[1].header)

# read the FITS table as a record array
table = hdulist[1].data
signal = table.timeline
nsamples = signal.size

# create array freq mode 1 to Nyquist Mode
delta_freq = FREQ_SAMPLING / nsamples
freq = delta_freq * np.arange(1, nsamples // 2 + 1)

# remove mode 0
spectrum = spectrum_estimate(signal)[1:]
