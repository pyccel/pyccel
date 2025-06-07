import numpy as np
from scipy.fftpack import fft, ifft, rfft, irfft

def my_fft(x: 'complex128[:]', n: 'int'):
    return fft(x, n)

def my_ifft(x: 'complex128[:]', n: 'int'):
    return ifft(x, n)

def my_rfft(x: 'float64[:]', n: 'int'):
    return rfft(x, n)

def my_irfft(x: 'float64[:]', n: 'int'):
    return irfft(x, n)

