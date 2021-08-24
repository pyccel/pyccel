#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
Module exposing the lapack library functions to pyccel (see http://www.netlib.org/lapack/)
"""

#$ header metavar ignore_at_import=True

from pyccel.stdlib.internal.lapack import dgetrf
from pyccel.stdlib.internal.lapack import dgetrs
from pyccel.stdlib.internal.lapack import dgbtrf
from pyccel.stdlib.internal.lapack import dgbtrs



#$ header macro (double(ab.shape[0], ab.shape[1]), int(ab.shape[0]),int) :: (ab, IPIV, info), dgetrf(ab) := dgetrf(ab.shape[0], ab.shape[1], ab, ab.shape[0], IPIV, info)
#$ header macro (double(b.shape[0], b.shape[1]), int) :: (b, info), dgetrs(ab, piv, b, s='N') := dgetrs(s, ab.shape[1], 1, ab, ab.shape[0], piv, b, b.count, info)

#$ header macro (double(ab.shape[0], ab.shape[1]), int(ab.shape[0]),int) :: (ab, IPIV, info), dgbtrf(ab, lab, uab) := dgbtrf(ab.shape[0], ab.shape[1], lab, uab, ab, ab.shape[0], IPIV, info)
#$ header macro (double(b.shape[0], b.shape[1]), int) :: (b, info), dgbtrs(ab, lab, uab, b, piv, s='N') := dgbtrs(s, ab.shape[1], lab, uab, 1, ab, ab.shape[0], piv, b, b.count, info)


