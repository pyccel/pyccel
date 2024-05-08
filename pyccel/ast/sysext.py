# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the sys module understood by pyccel
"""
from .core import PyccelFunctionDef, Module
from .internals import PyccelFunction
from .datatypes import VoidType
from .internals import LiteralInteger

__all__ = (
    'SysExit',
    'sys_constants',
    'sys_funcs',
    'sys_mod',
)

class SysExit(PyccelFunction):
    """
    Represents a call to sys.exit.

    Represents a call to sys.exit.

    Parameters
    ----------
    status : TypedAstNode, optional
        If status.dtype is NativeInteger it will be used as the exit_code,
        else the status will be printed to the stderror.
    """
    __slots__ = ()
    name      = 'exit'
    _class_type     = VoidType()
    _shape     = None

    def __init__(self, status=LiteralInteger(0)):
        super().__init__(status)

    @property
    def status(self):
        """return the arg of exit"""
        return self._args[0]

    def __str__(self):
        return f'exit({str(self.status)})'

sys_constants = {

}

sys_funcs = {
    'exit' : PyccelFunctionDef('exit', SysExit),
}

sys_mod = Module('sys',
        variables = (sys_constants.values()),
        funcs = (sys_funcs.values()),
        imports = [
        ])
