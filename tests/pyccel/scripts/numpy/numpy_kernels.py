# pylint: disable=missing-function-docstring, missing-module-docstring/
#
# Testing various kernels, adapted from
# https://github.com/redmod-team/profit/blob/master/profit/sur/backend/kernels.py
#

import numpy as np
from pyccel.decorators import types

@types('real[:]', 'real[:]', 'real')
def kern_sqexp(x0, x1, h):
    """Squared exponential kernel"""
    return np.real(np.exp(-0.5*np.sum(((x1 - x0)/h)**2)))


@types('real[:]', 'real[:]', 'real[:]')
def kern_sqexp_multiscale(x0, x1, h):
    """Squared exponential kernel with different scale in each direction"""
    return np.real(np.exp(-0.5*np.sum(((x1 - x0)/h)**2)))


@types('real[:]', 'real[:]', 'real')
def kern_wendland4(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3"""
    r = np.real(np.sqrt(np.sum(((x1 - x0)/h)**2)))
    if r < 1.0:
        ret = np.abs((1.0 - r**4)*(1.0 + 4.0*r))
    else:
        ret = 0.0
    return ret


@types('real[:]', 'real[:]', 'real[:]')
def kern_wendland4_multiscale(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3,
       different scale in each direction"""
    r = np.real(np.sqrt(np.sum(((x1 - x0)/h)**2)))
    if r < 1.0:
        ret = np.abs((1.0 - r**4)*(1.0 + 4.0*r))
    else:
        ret = 0.0
    return ret


@types('real[:,:]', 'real[:,:]', 'real[:]', 'real[:,:]')
def gp_matrix(x0, x1, a, K):
    """Constructs GP covariance matrix between two point tuples x0 and x1"""
    n0 = len(x0)
    n1 = len(x1)
    for k0 in range(n0):
        for k1 in range(n1):
            K[k0, k1] = a[1]*kern_sqexp(x0[k0, :], x1[k1, :], a[0])
