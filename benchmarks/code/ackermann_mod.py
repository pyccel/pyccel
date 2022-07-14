# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing functions for testing the ackerman algorithm using pyccel or pythran
"""

# pythran export ackermann(int,int)
def ackermann(m : int, n : int) -> int:
    """  Total computable function that is not primitive recursive.
    This function is useful for testing recursion
    """
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))
