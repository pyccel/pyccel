# -*- coding: utf-8 -*-

from sympy import Tuple
from sympy.core.basic import Basic

from pyccel.ast.core import FunctionCall
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import Variable
from pyccel.ast.core import Assign
from pyccel.ast.core import Return
from pyccel.ast.core import Module

#=======================================================================================

class F2PY_FunctionInterface(Basic):

    def __new__(cls, func, f2py_module_name, parent):
        if not(isinstance(func, FunctionDef) and func.is_static):
            raise TypeError('Wrong arguments')

        return Basic.__new__(cls, func, f2py_module_name, parent)

    @property
    def f2py_function(self):
        return self.args[0]

    @property
    def f2py_module_name(self):
        return self.args[1]

    @property
    def parent(self):
        return self.args[2]

#=======================================================================================

class F2PY_ModuleInterface(Basic):

    def __new__(cls, module, parents):
        if not isinstance(module, Module):
            raise TypeError('Expecting a Module')

        if not isinstance(parents, dict):
            raise TypeError('Expecting a dict')

        obj = Basic.__new__(cls, module)
        obj._parents = parents
        return obj

    @property
    def module(self):
        return self.args[0]

    @property
    def parents(self):
        return self._parents

#=======================================================================================

def as_static_function(func):
    assert(isinstance(func, FunctionDef))

    args    = func.arguments
    results = func.results
    body    = func.body
    arguments_inout = func.arguments_inout
    _results = []
    if results:
        if len(results) == 1:
            result = results[0]
            if result.rank > 0:
                # updates args
                args = list(args) + [result]
                arguments_inout += [False]
            else:
                _results = results

        else:
            raise NotImplementedError('when len(results) > 1')

    name = 'f2py_{}'.format(func.name).lower()

    # ...
    results_names = [i.name for i in results]
    _args = []
    _arguments_inout = []
    for i_a,a in enumerate(args):
        if not isinstance( a, Variable ):
            raise TypeError('Expecting a Variable type for {}'.format(a))

        rank = a.rank
        if rank > 0:
            # ...
            additional_args = []
            for i in range(0, rank):
                n_name = 'n{i}_{name}'.format(name=str(a.name), i=i)
                n_arg  = Variable('int', n_name)

                additional_args += [n_arg]

            shape_new = Tuple(*additional_args, sympify=False)
            # ...

            _args += additional_args
            for j in additional_args:
                _arguments_inout += [False]

            a_new = Variable( a.dtype, a.name,
                              allocatable = a.allocatable,
                              is_pointer  = a.is_pointer,
                              is_target   = a.is_target,
                              is_optional = a.is_optional,
                              shape       = shape_new,
                              rank        = a.rank,
                              order       = a.order,
                              precision   = a.precision)

            if not( a.name in results_names ):
                _args += [a_new]

            else:
                _results += [a_new]

        else:
            _args += [a]

        intent = arguments_inout[i_a]
        _arguments_inout += [intent]

    args = _args
    results = _results
    arguments_inout = _arguments_inout
    # ...

    return FunctionDef( name, list(args), results, body,
                        local_vars = func.local_vars,
                        is_static = True,
                        arguments_inout = arguments_inout )

