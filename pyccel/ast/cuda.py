#!/usr/bin/python
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
    'IndexedFunctionCall'
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

    args : tuple
        The arguments being passed to the function.

    numBlocks : NativeInteger
        The number of blocks.

    tpblock : NativeInteger
        The number of threads per block.

    launch_config : tuple
        Launch configuration of kernel call.

    current_function : FunctionDef, default: None
        The function where the call takes place.
    """
    __slots__ = ('_numBlocks','_tpblock','_func', '_args', '_launch_config')
    _attribute_nodes = (*FunctionCall._attribute_nodes, '_numBlocks', '_tpblock', '_launch_config')
    def __init__(self, func, args, numBlocks, tpblock, launch_config,current_function=None):
        self._numBlocks = numBlocks
        self._tpblock = tpblock
        self._launch_config = launch_config
        super().__init__(func, args, current_function)

    @property
    def numBlocks(self):
        """
        The number of blocks in the kernel being called.
    
        The number of blocks in the kernel being called.
        """
        return self._numBlocks

    @property
    def tpblock(self):
        """
        The number of threads per block.

        Launch configuration of kernel call.
        """
        return self._tpblock

    @property
    def launch_config(self):
        """
        Launch configuration of kernel call.

        Launch configuration of kernel call.
        """
        return self._launch_config

class IndexedFunctionCall(FunctionCall):
    """
    Represents an indexed function call in the code.
    
    class represents indexed function calls, encapsulating all
    relevant information for such calls within the codebase.

    Parameters
    ----------
    func : FunctionDef
        The function being called.

    args : tuple
        The arguments passed to the function.

    launch_config : tuple
        Launch configuration of kernel call.
    """
    __slots__ = ('_launch_config',)
    def __init__(self, func, args, launch_config):
        self._arguments = tuple(args)
        self._func_name = func
        self._launch_config = launch_config
        super().__init__(func, args, None)

    @property
    def launch_config(self):
        """
        Launch configuration of kernel call.
        
        Launch configuration of kernel call.
        """
        return self._launch_config
