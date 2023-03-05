#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the scipy module understood by pyccel
"""
from numpy import pi
from .core import Module, Import
from .variable import Constant

__all__ = ('scipy_mod', 'scipy_pi_const')

scipy_pi_const = Constant('float', 'pi', value=pi)

scipy_mod = Module('scipy',
        variables = (scipy_pi_const,),
        funcs = (),
        imports = [
            Import('constants',Module('constants', (scipy_pi_const,), ()))
            ])
