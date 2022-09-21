#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the sys module understood by pyccel
"""
from .core import PyccelFunctionDef, Module
from .internals import PyccelInternalFunction
from .datatypes import NativeVoid, NativeInteger
from .internals import LiteralInteger

class SysExit(PyccelInternalFunction):
    """Represents a call to  sys.exit

    arg : LiteralInteger, PyccelUnarySub(LiteralInteger)
    """
    __slots__ = ()
    name      = 'exit'
    _dtype     = NativeVoid()
    _precision = -1
    _rank      = 0
    _shape     = None
    _order     = None

    def __init__(self, arg=None):
        if arg is None:
            arg = LiteralInteger(0)
        super().__init__(arg)

    @property
    def arg(self):
        """return the arg of exit"""
        return self._args[0]

    def __str__(self):
        return f'exit({str(self.arg)})'

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
