# pylint: disable=missing-function-docstring, missing-module-docstring
a0 = [6]*10
a1 = sum(a0[i] for i in range(len(a0)))
a2 = sum(i for i in a0)
a3 = max(i if i>k else k for i in range(5) for k in range(10))
a4 = min(k if i>k else 0 if i==k else i for i in range(5) for k in range(10))

#$ header function incr(int)
def incr(x):
    y = x + 1
    return y
from numpy import ones

a=ones((5,5,5,5),'double')
b=ones((5),'int')

a5 = (2*sum(b[i] for i in range(5))**5+5)*min(j+1. for j in b)**4+9
a6  = 5+incr(2+incr(6+sum(b[i] for i in range(5))))
a7 = sum(sum(sum(a[i,k,o,2] for i in range(5)) for k in range(5)) for o in range(5))
a8 = min(min(sum(min(max(a[i,k,o,l]*l for i in range(5)) for k in range(5)) for o in range(5)) for l in range(5)),0.)
a9 = sum(sum(a[i,k,4,2] for i in range(5)) for k in range(5))**2

print(a1,a2,a3,a4,a5,a6,a7,a8,a9)
