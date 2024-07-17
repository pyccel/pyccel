#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This submodule contains cuda_arrays methods for Pyccel.
"""

def host_empty(shape):
    """
    Create an empty array on the host.

    Create an empty array on the host.

    Parameters
    ----------
    shape : tuple of int or int
        The shape of the array.

    Returns
    -------
    array
        The empty array on the host.
    """
    import numpy as np
    a = np.empty(shape)
    return a


