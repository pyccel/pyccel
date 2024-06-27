#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
CUDA Extension Module
Provides CUDA functionality for code generation.
"""
from .internals      import PyccelFunction

from .datatypes      import VoidType
from .core           import Module, PyccelFunctionDef

__all__ = (
    'CudaSynchronize',
)

class CudaSynchronize(PyccelFunction):
    """
    Represents a call to Cuda.synchronize for code generation.

    This class serves as a representation of the Cuda.synchronize method.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _shape     = None
    _class_type = VoidType()
    def __init__(self):
        super().__init__()

cuda_funcs = {
    'synchronize'       : PyccelFunctionDef('synchronize' , CudaSynchronize),
}

cuda_mod = Module('cuda',
    variables=[],
    funcs=cuda_funcs.values(),
    imports=[]
)

