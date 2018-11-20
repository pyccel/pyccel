# -*- coding: utf-8 -*-

from sympy import Tuple
from sympy.core.basic import Basic

#=======================================================================================
class F2PY_Function(Basic):

    def __new__(cls, func, module_name):
        return Basic.__new__(cls, func, module_name)

    @property
    def func(self):
        return self.args[0]

    @property
    def module_name(self):
        return self.args[1]

    @property
    def name(self):
        return 'f2py_{}'.format(self.func.name).lower()

#=======================================================================================

# module_name and name are different here
# module_name is the original module name
# name is the name we are giving for the new module
class F2PY_Module(Basic):

    def __new__(cls, functions, module_name):
        if not isinstance(functions, (tuple, list, Tuple)):
            raise TypeError('Expecting an iterable')

        functions = [F2PY_Function(f, module_name) for f in functions]
        return Basic.__new__(cls, functions, module_name)

    @property
    def functions(self):
        return self.args[0]

    @property
    def module_name(self):
        return self.args[1]

    @property
    def name(self):
        return 'f2py_{}'.format(self.module_name).lower()

#=======================================================================================

# this is used as a python interface for a F2PY_Function
# it takes a F2PY_Function as input
class F2PY_FunctionInterface(Basic):

    def __new__(cls, func, f2py_module_name):
        if not isinstance(func, F2PY_Function):
            raise TypeError('Expecting a F2PY_Function')

        return Basic.__new__(cls, func, f2py_module_name)

    @property
    def f2py_function(self):
        return self.args[0]

    @property
    def f2py_module_name(self):
        return self.args[1]

#=======================================================================================

class F2PY_ModuleInterface(Basic):

    def __new__(cls, module):
        if not isinstance(module, F2PY_Module):
            raise TypeError('Expecting a F2PY_Module')

        return Basic.__new__(cls, module)

    @property
    def module(self):
        return self.args[0]

