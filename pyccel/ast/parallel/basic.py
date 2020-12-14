# coding: utf-8
#--------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See file LICENSE or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. #
#--------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring, missing-module-docstring/
from sympy.core.basic import Basic as sm_Basic

__all__ = ('Basic',)

#==============================================================================
class Basic(sm_Basic):
    is_integer = False
    _dtypes = {}
    _dtypes['size'] = 'int'
    _dtypes['rank'] = 'int'

    def __new__(cls, *args, **options):
        return super(Basic, cls).__new__(cls, *args, **options)

    def dtype(self, attr):
        """Returns the datatype of a given attribut/member."""
        return self._dtypes[attr]
