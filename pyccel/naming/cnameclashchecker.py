# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in C
"""
from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.strings import create_incremented_string

class CNameClashChecker(metaclass = Singleton):
    """ Class containing functions to help avoid problematic names in C
    """
    # Keywords as mentioned on https://en.cppreference.com/w/c/keyword
    keywords = set(['isign', 'fsign', 'csign', 'auto', 'break', 'case', 'char', 'const',
        'continue', 'default', 'do', 'double', 'else', 'enum',
        'extern', 'float', 'for', 'goto', 'if', 'inline', 'int',
        'long', 'register', 'restrict', 'return', 'short', 'signed',
        'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
        'unsigned', 'void', 'volatile', 'whie', '_Alignas',
        '_Alignof', '_Atomic', '_Bool', '_Complex', 'Decimal128',
        '_Decimal32', '_Decimal64', '_Generic', '_Imaginary',
        '_Noreturn', '_Static_assert', '_Thread_local', 't_ndarray',
        'array_create', 'new_slice', 'array_slicing', 'alias_assign',
        'transpose_alias_assign', 'array_fill', 't_slice',
        'GET_INDEX_EXP1', 'GET_INDEX_EXP2', 'GET_INDEX_EXP2',
        'GET_INDEX_EXP3', 'GET_INDEX_EXP4', 'GET_INDEX_EXP5',
        'GET_INDEX_EXP6', 'GET_INDEX_EXP7', 'GET_INDEX_EXP8',
        'GET_INDEX_EXP9', 'GET_INDEX_EXP10', 'GET_INDEX_EXP11',
        'GET_INDEX_EXP12', 'GET_INDEX_EXP13', 'GET_INDEX_EXP14',
        'GET_INDEX_EXP15', 'NUM_ARGS_H1', 'NUM_ARGS',
        'GET_INDEX_FUNC_H2', 'GET_INDEX_FUNC', 'GET_INDEX',
        'INDEX', 'GET_ELEMENT', 'free_array', 'free_pointer',
        'get_index', 'numpy_to_ndarray_strides',
        'numpy_to_ndarray_shape', 'get_size'])

    def has_clash(self, name, symbols):
        """ Indicate whether the proposed name causes any clashes
        """
        return any(name == k for k in self.keywords) or \
               any(name == s for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """ Get the name that will be used in the fortran code
        """
        if len(name)>4 and all(name[i] == '_' for i in (0,1,-1,-2)):
            # Ignore magic methods
            return name
        if name[0] == '_':
            name = 'private'+name
        prefix = name
        coll_symbols = self.keywords.copy()
        coll_symbols.update(s.lower() for s in symbols)
        if prefix in coll_symbols:
            counter = 1
            new_name, counter = create_incremented_string(coll_symbols,
                    prefix = prefix, counter = counter)
            name = name+new_name[-5:]
        return name

