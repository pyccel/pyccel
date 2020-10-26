# coding: utf-8

from .datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool

__all__ = ('str_dtype',)

def str_dtype(dtype):

    """
    This function takes a datatype and returns a sympy datatype as a string

    Example
    -------
    >>> str_dtype('int')
    'integer'
    >>> str_dtype(NativeInteger())
    'integer'

    """
    if isinstance(dtype, str):
        if dtype == 'int':
            return 'integer'
        elif dtype== 'real':
            return 'real'
        else:
            return dtype
    if isinstance(dtype, NativeInteger):
        return 'integer'
    elif isinstance(dtype, NativeReal):
        return 'real'
    elif isinstance(dtype, NativeComplex):
        return 'complex'
    elif isinstance(dtype, NativeBool):
        return 'bool'
    else:
        raise TypeError('Unknown datatype {0}'.format(str(dtype)))
