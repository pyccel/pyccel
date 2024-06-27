# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
CUDA Module
This module provides a collection of classes and utilities for CUDA programming.
"""
from pyccel.ast.core import FunctionCall

__all__ = (
    'KernelCall',
)

class KernelCall(FunctionCall):
    """
    Represents a kernel function call in the code.

    The class serves as a representation of a kernel
    function call within the codebase.

    Parameters
    ----------
    func : FunctionDef
        The definition of the function being called.

    args : iterable of FunctionCallArgument
        The arguments passed to the function.

    num_blocks : TypedAstNode
        The number of blocks. These objects must have a primitive type of `PrimitiveIntegerType`.

    tp_block : TypedAstNode
        The number of threads per block. These objects must have a primitive type of `PrimitiveIntegerType`.

    current_function : FunctionDef, optional
        The function where the call takes place.
    """
    __slots__ = ('_num_blocks','_tp_block')
    _attribute_nodes = (*FunctionCall._attribute_nodes, '_num_blocks', '_tp_block')

    def __init__(self, func, args, num_blocks, tp_block, current_function = None):
        self._num_blocks = num_blocks
        self._tp_block = tp_block
        super().__init__(func, args, current_function)

    @property
    def num_blocks(self):
        """
        The number of blocks in the kernel being called.

        The number of blocks in the kernel being called.
        """
        return self._num_blocks

    @property
    def tp_block(self):
        """
        The number of threads per block.

        The number of threads per block.
        """
        return self._tp_block

