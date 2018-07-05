#$ header function incr(int)
def incr(x):
    x = x + 1

from numpy import ones

s=ones((5,5,5,5),'double')
b=ones((5),'int')

a=(2*sum(b[i] for i in range(5))**5+5)*min(j+1. for j in b)**4+9
incr(sum(b[i] for i in range(5)))
m=sum(sum(sum(s[i,k,o,2] for i in range(5)) for k in range(5)) for o in range(5))
nm=min(min(sum(min(max(s[i,k,o,l]*l for i in range(5)) for k in range(5)) for o in range(5)) for l in range(5)),0.)
knm=sum(sum(s[i,k,4,2] for i in range(5)) for k in range(5))**2

print a,m,nm,knm

