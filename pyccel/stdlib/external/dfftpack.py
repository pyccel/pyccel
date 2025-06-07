#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module exposing the dfftpack library functions to pyccel (see https://www.netlib.org/fftpack/)
"""
import numpy as np
from pyccel.decorators import inline

#$ header metavar print=True

from pyccel.stdlib.internal.dfftpack import dffti
from pyccel.stdlib.internal.dfftpack import dfftf
from pyccel.stdlib.internal.dfftpack import dfftb

from pyccel.stdlib.internal.dfftpack import zffti
from pyccel.stdlib.internal.dfftpack import zfftf
from pyccel.stdlib.internal.dfftpack import zfftb


@inline
def fft(x : 'float[:]|complex[:]', n : int = None):
    if n is None:
        n = x.size
    w = np.empty(4*n+15)
    y = np.empty(x.shape, dtype=complex)
    y[:] = x[:]
    zffti(np.int32(n),w)
    zfftf(np.int32(n),y,w)
    return y

@inline
def ifft(x : 'float[:]|complex[:]', n : int = None):
    if n is None:
        n = x.size
    w = np.empty(4*n+15)
    y = np.empty(x.shape, dtype=complex)
    y[:] = x[:]
    zffti(np.int32(n), w)
    zfftb(np.int32(n), y, w)
    return y

@inline
def rfft(x : 'float[:]', n : int = None):
    if n is None:
        n = x.size
    w = np.empty(2*n+15)
    y = np.array(x)
    dffti(np.int32(n),w)
    dfftf(np.int32(n),y,w)
    return y

@inline
def irfft(x : 'float[:]', n : int = None):
    if n is None:
        n = x.size
    w = np.empty(2*n+15)
    y = np.array(x)
    dffti(np.int32(n),w)
    dfftb(np.int32(n),y,w)
    return y
