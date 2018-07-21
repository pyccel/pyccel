#$ header metavar ignore_at_import=True

from pyccel.stdlib.internal.lapack import dgetrf
from pyccel.stdlib.internal.lapack import dgetrs



#$ header macro (ab, IPIV, info), dgetrf(ab) := dgetrf(ab.shape[0], ab.shape[1], ab, ab.shape[0], IPIV, info)
#$ header macro (ab, info), dgetrs(ab,piv,b,s='N') := dgetrs(s,ab.shape[1],1 , ab, ab.shape[0], piv,b,b.count, info)
 
