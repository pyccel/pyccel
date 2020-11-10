#$ header metavar ignore_at_import=True

from pyccel.stdlib.internal.lapack import dgetrf
from pyccel.stdlib.internal.lapack import dgetrs
from pyccel.stdlib.internal.lapack import dgbtrf
from pyccel.stdlib.internal.lapack import dgbtrs



#$ header macro (ab, IPIV, info), dgetrf(ab) := dgetrf(ab.shape[0], ab.shape[1], ab, ab.shape[0], IPIV, info)
#$ header macro (b, info), dgetrs(ab, piv, b, s='N') := dgetrs(s, ab.shape[1], 1, ab, ab.shape[0], piv, b, b.count, info)

#$ header macro (ab, IPIV, info), dgbtrf(ab, lab, uab) := dgbtrf(ab.shape[1], ab.shape[1], lab, uab, ab, ab.shape[0], IPIV, info)
#$ header macro (b, info), dgbtrs(ab, lab, uab, b, piv, s='N') := dgbtrs(s, ab.shape[1], lab, uab, 1, ab, ab.shape[0], piv, b, b.count, info)


