#$ header metavar print=True

from pyccel.stdlib.internal.dfftpack import dffti
from pyccel.stdlib.internal.dfftpack import dfftf
from pyccel.stdlib.internal.dfftpack import dfftb

from pyccel.stdlib.internal.dfftpack import zffti
from pyccel.stdlib.internal.dfftpack import zfftf
from pyccel.stdlib.internal.dfftpack import zfftb



#$ header function fft(double[:]|complex[:], complex[:], int)
def fft(x, y, n):
    from numpy import empty
    w = empty(4*n+15)
    y[:] = x[:]
    zffti(n,w)
    zfftf(n,y,w)

#$ header function ifft(double[:]|complex[:], complex[:], int)
def ifft(x, y, n):
    from numpy import empty
    w = empty(4*n+15)
    y[:] = x[:]
    zffti(n, w)
    zfftb(n, x, w)

#$ header function rfft(double[:], double[:], int)
def rfft(x, y, n):
    from numpy import empty
    w = empty(2*n+15)
    y[:] = x[:]
    dffti(n,w)
    dfftf(n,y,w)

#$ header function irfft(double[:], double[:], int)
def irfft(x, y, n):
    from numpy import empty
    w = empty(2*n+15)
    y[:] = x[:]
    dffti(n,w)
    dfftb(n,y,w)



#$ header macro (y), fft (x, n=x.count) := fft (x, y, n)
#$ header macro (y), ifft(x, n=x.count) := ifft(x, y, n)

#$ header macro (y), rfft (x,n=x.count) := rfft (x, y, n)
#$ header macro (y), irfft(x,n=x.count) := irfft(x, y, n)


