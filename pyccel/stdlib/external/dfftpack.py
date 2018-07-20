from pyccel.stdlib.internal.dfftpack import dffti
from pyccel.stdlib.internal.dfftpack import dfftf
from pyccel.stdlib.internal.dfftpack import dfftb

from pyccel.stdlib.internal.dfftpack import dzffti
from pyccel.stdlib.internal.dfftpack import dzfftf
from pyccel.stdlib.internal.dfftpack import dzfftb

from pyccel.stdlib.internal.dfftpack import dcosqi
from pyccel.stdlib.internal.dfftpack import dcosqf
from pyccel.stdlib.internal.dfftpack import dcosqb
from pyccel.stdlib.internal.dfftpack import dcosti
from pyccel.stdlib.internal.dfftpack import dcost

from pyccel.stdlib.internal.dfftpack import dsinqi
from pyccel.stdlib.internal.dfftpack import dsinqf
from pyccel.stdlib.internal.dfftpack import dsinqb
from pyccel.stdlib.internal.dfftpack import dsinti
from pyccel.stdlib.internal.dfftpack import dsint

from pyccel.stdlib.internal.dfftpack import zffti
from pyccel.stdlib.internal.dfftpack import zfftf
from pyccel.stdlib.internal.dfftpack import zfftb	

@types(double[:],int)
def fft_real(x,n):
    from numpy import empty
    w = empty(2*n+15)
    dffti(n,w)
    dfftf(n,x,w)

@types(complex[:],int)
def fft_complex(x,n):
    from numpy import empty
    w = empty(4*n+15)
    zffti(n,w)
    zfftf(n,x,w)


@types(double[:],int)
def ifft_real(x,n):
    from numpy import empty
    w = empty(2*n+15)
    dffti(n,w)
    dfftb(n,x,w)

@types(complex[:],int)
def ifft_complex(x,n):
    from numpy import empty
    w = empty(4*n+15)
    zffti(n,w)
    zfftb(n,x,w)

