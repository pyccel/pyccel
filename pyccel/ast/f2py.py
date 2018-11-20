# -*- coding: utf-8 -*-

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
        return 'f2py_{}'.format(self.func.name)

class F2PY_Module(Basic):

    def __new__(cls, module):
        return Basic.__new__(cls, module)

    @property
    def module(self):
        return self.args[0]
