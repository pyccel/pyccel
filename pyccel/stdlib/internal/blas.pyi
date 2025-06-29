"""
Pyccel header for BLAS.
"""
#$ header metavar module_version='3.8'
#$ header metavar ignore_at_import=True
#$ header metavar save=True
#$ header metavar libraries='blas'
#$ header metavar external=True
from numpy import float32, int32, float64


# .......................................
#             LEVEL-1
# .......................................

def srotg(anon_0 : 'float32', anon_1 : 'float32', anon_2 : 'float32', anon_3 : 'float32') -> None:
    ...

def drotg(anon_0 : 'float64', anon_1 : 'float64', anon_2 : 'float64', anon_3 : 'float64') -> None:
    ...

def srotmg(anon_0 : 'float32', anon_1 : 'float32', anon_2 : 'float32', anon_3 : 'float32', anon_4 : 'float32[:]') -> None:
    ...

def drotmg(anon_0 : 'float64', anon_1 : 'float64', anon_2 : 'float64', anon_3 : 'float64', anon_4 : 'float64[:]') -> None:
    ...

def srot(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32', anon_6 : 'float32') -> None:
    ...

def drot(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64', anon_6 : 'float64') -> None:
    ...

def srotm(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:]') -> None:
    ...

def drotm(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:]') -> None:
    ...

def sswap(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32', anon_3 : 'float32[:]', anon_4 : 'int32') -> None:
    ...

def dswap(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32') -> None:
    ...

def sscal(anon_0 : 'int32', anon_1 : 'float32', anon_2 : 'float32[:]', anon_3 : 'int32') -> None:
    ...

def dscal(anon_0 : 'int32', anon_1 : 'float64', anon_2 : 'float64[:]', anon_3 : 'int32') -> None:
    ...

def sasum(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32') -> 'float32':
    ...

def dasum(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32') -> 'float64':
    ...

def isamax(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32') -> 'int32':
    ...

def idamax(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32') -> 'int32':
    ...

def saxpy(anon_0 : 'int32', anon_1 : 'float32', anon_2 : 'float32[:]', anon_3 : 'int32', anon_4 : 'float32[:]', anon_5 : 'int32') -> None:
    ...

def daxpy(anon_0 : 'int32', anon_1 : 'float64', anon_2 : 'float64[:]', anon_3 : 'int32', anon_4 : 'float64[:]', anon_5 : 'int32') -> None:
    ...

def scopy(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32', anon_3 : 'float32[:]', anon_4 : 'int32') -> None:
    ...

def dcopy(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32') -> None:
    ...

def sdot(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32', anon_3 : 'float32[:]', anon_4 : 'int32') -> 'float32':
    ...

def ddot(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32') -> 'float64':
    ...

def sdsdot(anon_0 : 'int32', anon_1 : 'float32', anon_2 : 'float32[:]', anon_3 : 'int32', anon_4 : 'float32[:]', anon_5 : 'int32') -> 'float32':
    ...

def dsdot(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32', anon_3 : 'float64[:]', anon_4 : 'int32') -> 'float64':
    ...

def snrm2(anon_0 : 'int32', anon_1 : 'float32[:]', anon_2 : 'int32') -> 'float32':
    ...

def dnrm2(anon_0 : 'int32', anon_1 : 'float64[:]', anon_2 : 'int32') -> 'float64':
    ...
# .......................................

# .......................................
#             LEVEL-2
# .......................................
def sgemv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'float32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float32[:]', anon_7 : 'int32', anon_8 : 'float32', anon_9 : 'float32[:]', anon_10 : 'int32') -> None:
    ...

def dgemv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'float64', anon_4 : 'float64[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float64[:]', anon_7 : 'int32', anon_8 : 'float64', anon_9 : 'float64[:]', anon_10 : 'int32') -> None:
    ...

def sgbmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float32', anon_6 : 'float32[:,:](order=F)', anon_7 : 'int32', anon_8 : 'float32[:]', anon_9 : 'int32', anon_10 : 'float32', anon_11 : 'float32[:]', anon_12 : 'int32') -> None:
    ...

def dgbmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float64', anon_6 : 'float64[:,:](order=F)', anon_7 : 'int32', anon_8 : 'float64[:]', anon_9 : 'int32', anon_10 : 'float64', anon_11 : 'float64[:]', anon_12 : 'int32') -> None:
    ...

def ssymv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:,:](order=F)', anon_4 : 'int32', anon_5 : 'float32[:]', anon_6 : 'int32', anon_7 : 'float32', anon_8 : 'float32[:]', anon_9 : 'int32') -> None:
    ...

def dsymv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:,:](order=F)', anon_4 : 'int32', anon_5 : 'float64[:]', anon_6 : 'int32', anon_7 : 'float64', anon_8 : 'float64[:]', anon_9 : 'int32') -> None:
    ...

def ssbmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'float32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float32[:]', anon_7 : 'int32', anon_8 : 'float32', anon_9 : 'float32[:]', anon_10 : 'int32') -> None:
    ...

def dsbmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'float64', anon_4 : 'float64[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float64[:]', anon_7 : 'int32', anon_8 : 'float64', anon_9 : 'float64[:]', anon_10 : 'int32') -> None:
    ...

def sspmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:,:](order=F)', anon_4 : 'float32[:]', anon_5 : 'int32', anon_6 : 'float32', anon_7 : 'float32[:]', anon_8 : 'int32') -> None:
    ...

def dspmv(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:,:](order=F)', anon_4 : 'float64[:]', anon_5 : 'int32', anon_6 : 'float64', anon_7 : 'float64[:]', anon_8 : 'int32') -> None:
    ...

def strmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float32[:]', anon_7 : 'int32') -> None:
    ...

def dtrmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float64[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float64[:]', anon_7 : 'int32') -> None:
    ...

def stbmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float32[:]', anon_8 : 'int32') -> None:
    ...

def dtbmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float64[:]', anon_8 : 'int32') -> None:
    ...

def stpmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'float32[:]', anon_6 : 'int32') -> None:
    ...

def dtpmv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float64[:,:](order=F)', anon_5 : 'float64[:]', anon_6 : 'int32') -> None:
    ...

def strsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float32[:]', anon_7 : 'int32') -> None:
    ...

def dtrsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float64[:,:](order=F)', anon_5 : 'int32', anon_6 : 'float64[:]', anon_7 : 'int32') -> None:
    ...

def stbsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float32[:]', anon_8 : 'int32') -> None:
    ...

def dtbsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float64[:]', anon_8 : 'int32') -> None:
    ...

def stpsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float32[:,:](order=F)', anon_5 : 'float32[:]', anon_6 : 'int32') -> None:
    ...

def dtpsv(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'int32', anon_4 : 'float64[:,:](order=F)', anon_5 : 'float64[:]', anon_6 : 'int32') -> None:
    ...

def sger(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:]', anon_6 : 'int32', anon_7 : 'float32[:,:](order=F)', anon_8 : 'int32') -> None:
    ...

def dger(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:]', anon_6 : 'int32', anon_7 : 'float64[:,:](order=F)', anon_8 : 'int32') -> None:
    ...

def ssyr(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32') -> None:
    ...

def dsyr(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32') -> None:
    ...

def sspr(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:,:](order=F)') -> None:
    ...

def dspr(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:,:](order=F)') -> None:
    ...

def ssyr2(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:]', anon_6 : 'int32', anon_7 : 'float32[:]', anon_8 : 'int32') -> None:
    ...

def dsyr2(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:]', anon_6 : 'int32', anon_7 : 'float64[:]', anon_8 : 'int32') -> None:
    ...

def sspr2(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float32', anon_3 : 'float32[:]', anon_4 : 'int32', anon_5 : 'float32[:]', anon_6 : 'int32', anon_7 : 'float32[:]', anon_8 : 'int32') -> None:
    ...

def dspr2(anon_0 : 'str', anon_1 : 'int32', anon_2 : 'float64', anon_3 : 'float64[:]', anon_4 : 'int32', anon_5 : 'float64[:]', anon_6 : 'int32', anon_7 : 'float64[:]', anon_8 : 'int32') -> None:
    ...
# .......................................

# .......................................
#             LEVEL-3
# .......................................
def sgemm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float32', anon_6 : 'float32[:,:](order=F)', anon_7 : 'int32', anon_8 : 'float32[:,:](order=F)', anon_9 : 'int32', anon_10 : 'float32', anon_11 : 'float32[:,:](order=F)', anon_12 : 'int32') -> None:
    ...

def dgemm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'float64', anon_6 : 'float64[:,:](order=F)', anon_7 : 'int32', anon_8 : 'float64[:,:](order=F)', anon_9 : 'int32', anon_10 : 'float64', anon_11 : 'float64[:,:](order=F)', anon_12 : 'int32') -> None:
    ...

def ssymm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float32[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float32', anon_10 : 'float32[:,:](order=F)', anon_11 : 'int32') -> None:
    ...

def dsymm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float64', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float64[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float64', anon_10 : 'float64[:,:](order=F)', anon_11 : 'int32') -> None:
    ...

def ssyrk(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float32', anon_8 : 'float32[:,:](order=F)', anon_9 : 'int32') -> None:
    ...

def dsyrk(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float64', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float64', anon_8 : 'float64[:,:](order=F)', anon_9 : 'int32') -> None:
    ...

def ssyr2k(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float32', anon_5 : 'float32[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float32[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float32', anon_10 : 'float32[:,:](order=F)', anon_11 : 'int32') -> None:
    ...

def dsyr2k(anon_0 : 'str', anon_1 : 'str', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'float64', anon_5 : 'float64[:,:](order=F)', anon_6 : 'int32', anon_7 : 'float64[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float64', anon_10 : 'float64[:,:](order=F)', anon_11 : 'int32') -> None:
    ...

def strmm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'str', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'float32', anon_7 : 'float32[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float32[:,:](order=F)', anon_10 : 'int32') -> None:
    ...

def dtrmm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'str', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'float64', anon_7 : 'float64[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float64[:,:](order=F)', anon_10 : 'int32') -> None:
    ...

def strsm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'str', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'float32', anon_7 : 'float32[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float32[:,:](order=F)', anon_10 : 'int32') -> None:
    ...

def dtrsm(anon_0 : 'str', anon_1 : 'str', anon_2 : 'str', anon_3 : 'str', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'float64', anon_7 : 'float64[:,:](order=F)', anon_8 : 'int32', anon_9 : 'float64[:,:](order=F)', anon_10 : 'int32') -> None:
    ...
