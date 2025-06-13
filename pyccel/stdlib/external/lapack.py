#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
Module exposing the lapack library functions to pyccel (see http://www.netlib.org/lapack/)
"""

#$ header metavar ignore_at_import=True
import numpy as np
from pyccel.decorators import inline
import pyccel.stdlib.internal.lapack as lapack

@inline
def dgetrf(a : 'float64[:,:]', overwrite_a : bool = False):
    ipiv = np.empty(np.int32(a.shape[1]), dtype = np.int32)
    info : np.int32
    if overwrite_a:
        lapack.dgetrf(np.int32(a.shape[1]), np.int32(a.shape[0]), a.T, np.int32(a.shape[1]), ipiv, info)
        return a, ipiv, info
    else:
        lu = np.array(a)
        lapack.dgetrf(np.int32(a.shape[1]), np.int32(a.shape[0]), lu.T, np.int32(a.shape[1]), ipiv, info)
        return lu, ipiv, info

@inline
def dgetrs(lu : 'float64[:,:]', piv : 'int32[:]', b : 'float64[:,:]',
           trans : str ='N', overwrite_b : bool = False):
    ipiv = np.empty(m, dtype = np.int32)
    info : np.int32
    if not overwrite_b:
        storage_b = np.array(b)
        b_ptr = storage_b.T
    else:
        b_ptr = b.T

    lapack.dgetrs(s, np.int32(lu.shape[0]), np.int32(b.shape[0]), lu.T,
            np.int32(lu.shape[1]), piv, b_ptr, np.int32(b.shape[1]), info)
    return b, info

@inline
def dgbtrf(ab : 'float64[:,:]', kl : int, ku : int,
        m : int = None, n : int = None, ldab : int = None,
        overwrite_ab : bool = False):
    m_val = np.int32(ab.shape[0] if m is None else m)
    n_val = np.int32(ab.shape[0] if n is None else n)
    ldab_val = np.int32(max(ab.shape[1], 1) if ldab is None else ldab)
    ipiv = np.empty(min(m, n), dtype = np.int32)
    info : np.int32
    if overwrite_ab:
        lapack.dgbtrf(m_val, n_val, np.int32(kl), np.int32(ku), ab.T, ldab_val, ipiv, info)
        return ab, ipiv, info
    else:
        lu = np.array(ab)
        lapack.dgbtrf(m_val, n_val, np.int32(kl), np.int32(ku), lu.T, ldab_val, ipiv, info)
        return lu, ipiv, info

@inline
def dgbtrs(ab : 'float64[:,:]', kl : int, ku : int, b : 'float64[:,:]',
           ipiv : np.int32, trans : str ='N', n : int = None,
           ldab : int = None, ldb : int = None, overwrite_b : bool = False):
    n_val = np.int32(ab.shape[0] if n is None else n)
    ldab_val = np.int32(max(ab.shape[1], 1) if ldab is None else ldab)
    ldb_val = np.int32(ab.shape[0] if ldb is None else ldb)
    info : np.int32
    lapack.dgbtrs(trans, n_val, kl, ku, b.shape[0], ab.T, ldab_val, ipiv, b.T, ldb_val, info)
    return b, info


