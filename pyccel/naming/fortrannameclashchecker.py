# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in Fortran.
"""
import warnings

from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.strings import create_incremented_string
from .languagenameclashchecker import LanguageNameClashChecker

class FortranNameClashChecker(LanguageNameClashChecker):
    """
    Class containing functions to help avoid problematic names in Fortran.

    A class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may be due to
    capitalisation (as Fortran is not case-sensitive), or due to the use of reserved
    keywords.
    """
    # Keywords as mentioned on https://fortranwiki.org/fortran/show/Keywords
    # Intrinsic functions as mentioned on https://pages.mtu.edu/~shene/COURSES/cs201/NOTES/chap02/funct.html
    keywords = set(['assign', 'backspace', 'block', 'blockdata',
            'call', 'close', 'common', 'continue', 'data',
            'dimension', 'do', 'else', 'elseif', 'end', 'endfile',
            'endif', 'endfunction', 'endmodule', 'endprogram',
            'endsubroutine', 'entry', 'equivalence', 'external',
            'format', 'function', 'goto', 'if', 'implicit',
            'intrinsic', 'open', 'parameter', 'pause', 'print',
            'program', 'read', 'return', 'rewind', 'rewrite',
            'save', 'stop', 'subroutine', 'then', 'write',
            'allocatable', 'allocate', 'case', 'contains', 'cycle',
            'deallocate', 'elsewhere', 'exit', 'include', 'interface',
            'intent', 'module', 'namelist', 'nullify', 'only',
            'operator', 'optional', 'pointer', 'private', 'procedure',
            'public', 'recursive', 'result', 'select', 'sequence',
            'target', 'use', 'while', 'where', 'elemental', 'forall',
            'pure', 'abstract', 'associate', 'asynchronous', 'bind',
            'class', 'deferred', 'enum', 'enumerator', 'extends',
            'final', 'flush', 'generic', 'import', 'non_overrideable',
            'nopass', 'pass', 'protected', 'value', 'volatile',
            'wait', 'codimension', 'concurrent', 'contiguous',
            'critical', 'error', 'submodule', 'sync', 'lock',
            'unlock', 'test', 'abs', 'sqrt', 'sin', 'cos', 'tan',
            'asin', 'acos', 'atan', 'exp', 'log', 'int', 'nint',
            'floor', 'fraction', 'real', 'max', 'mod', 'count',
            'pack', 'numpy_sign', 'c_associated', 'c_loc', 'c_f_pointer', 'c_ptr'])

    def has_clash(self, name, symbols):
        """ Indicate whether the proposed name causes any clashes
        """
        name = name.lower()
        return any(name == k for k in self.keywords) or \
               any(name == s.lower() for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """
        Get a valid name which doesn't collision with symbols or Fortran keywords.

        Find a new name based on the suggested name which will not cause
        conflicts with Fortran keywords, does not conflict with the provided symbols,
        and is a valid name in Fortran code.

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
        if len(name)>4 and all(name[i] == '_' for i in (0,1,-1,-2)):
            # Ignore magic methods
            return name
        if name[0] == '_':
            name = 'private'+name
        name = self._get_collisionless_name(name, symbols)
        if len(name) > 96:
            warnings.warn("Name {} is too long for Fortran. This may cause compiler errors".format(name))
        return name
