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
        '_Noreturn', '_Static_assert', '_Thread_local',
        'I', 'cspan_copy', 'c_foreach', 'c_COLMAJOR', 'c_ROWMAJOR', 'cspan_md_layout',
        'using_cspan', 'STC_CSPAN_INDEX_TYPE', 'array_int64_1d', 'array_int64_2d',
        'array_int64_3d', 'array_int32_1d', 'array_int32_2d', 'array_int32_3d',
        'array_float_1d', 'array_float_2d', 'array_float_3d', 'array_double_1d',
        'array_double_2d', 'array_double_3d', 'array_bool_1d', 'array_bool_2d',
        'array_bool_3d', 'array_float_complex_1d', 'array_float_complex_2d',
        'array_float_complex_3d', 'array_double_complex_1d', 'array_double_complex_2d',
        'array_double_complex_3d', 'c_ALL', 'c_END', 'cspan_slice', 'cspan_transpose',
        'complex_max', 'complex_min', 'expm1', 'complex_expm1', 'main'])

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

    def get_collisionless_name(self, name, symbols, *, prefix, context, parent_context):
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
        prefix : str
            The prefix that may be added to the name to provide context information.
        context : str
            The context where the name will be used.
        parent_context : str
            The type of the scope where the object with this name will be saved.

        Returns
        -------
        str
            A new name which is collision free.
        """
        assert context in ('module', 'function', 'class', 'variable', 'wrapper')
        assert parent_context in ('module', 'function', 'class', 'loop', 'program')
        if context == 'wrapper':
            # wrapper names are based off names which already have prefixes so there is no
            # need to add more
            return self._get_collisionless_name(name, symbols)
        if name == '__init__':
            name = 'create'
        if name == '__del__':
            name = 'drop'
        if len(name)>4 and all(name[i] == '_' for i in (0,1,-1,-2)):
            name = 'operator' + name[1:-2]
        if name[0] == '_':
            name = 'private'+name
        if context == 'function' or (parent_context == 'module' and context != 'module'):
            name = prefix + name
        return self._get_collisionless_name(name, symbols)

