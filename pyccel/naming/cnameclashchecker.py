# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in C
"""
from .languagenameclashchecker import LanguageNameClashChecker

class CNameClashChecker(LanguageNameClashChecker):
    """
    Class containing functions to help avoid problematic names in C.

    A class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may be due to
    new variables, or due to the use of reserved keywords.
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
        'numpy_to_ndarray_shape', 'get_size', 'order_f', 'order_c', 'array_copy_data',
        'I'])

    def has_clash(self, name, symbols):
        """
        Indicate whether the proposed name causes any clashes.

        Indicate whether the proposed name causes any clashes by comparing it with the
        reserved keywords and the symbols which are already defined in the scope.

        Parameters
        ----------
        name : str
            The proposed name.
        symbols : set of str
            The symbols already used in the scope.

        Returns
        -------
        bool
            True if the name clashes with an existing name. False otherwise.
        """
        return name in self.keywords or name in symbols

    def get_collisionless_name(self, name, symbols):
        """
        Get a valid name which doesn't collision with symbols or C keywords.

        Find a new name based on the suggested name which will not cause
        conflicts with C keywords, does not appear in the provided symbols,
        and is a valid name in C code.

        Parameters
        ----------
        name : str
            The suggested name.
        symbols : set
            Symbols which should be considered as collisions.

        Returns
        -------
        str
            A new name which is collision free.
        """
        if name in ('__init__', '__del__'):
            return name
        if len(name)>4 and all(name[i] == '_' for i in (0,1,-1,-2)):
            name = 'operator' + name[1:-2]
        if name[0] == '_':
            name = 'private'+name
        return self._get_collisionless_name(name, symbols)

