# -*- coding: utf-8 -*-

from sympy import Tuple
from sympy.core.basic import Basic

from pyccel.ast.core import FunctionCall
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import Variable
from pyccel.ast.core import Assign
from pyccel.ast.core import Return

#=======================================================================================
class F2PY_Function(FunctionDef):

    def __new__(cls, func, module_name):

        args    = func.arguments
        results = func.results
        arguments_inout = func.arguments_inout
        _results = []
        if results:
            if len(results) == 1:
                result = results[0]
                if result.rank > 0:
                    body = [FunctionCall(func, args)]
                    # updates args
                    args = list(args) + [result]
                    arguments_inout += [False]
                else:
                    body  = [Assign(result, FunctionCall(func, args))]
                    body += [Return(result)]
                    _results = results

            else:
                raise NotImplementedError('when len(results) > 1')
        else:
            body = [FunctionCall(func, args)]

        name = 'f2py_{}'.format(func.name)

        # ...

        results_names = [i.name for i in results]
        _args = []
        _arguments_inout = []
        for i_a,a in enumerate(args):
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

        obj = FunctionDef.__new__( cls, name, list(args), results, body,
                                   is_static = True,
                                   arguments_inout = arguments_inout )

        obj._func = func
        return obj

    @property
    def func(self):
        return self._func


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

    def __new__(cls, func, f2py_module_name, parent=None):
        if isinstance(func, F2PY_Function):
            parent = func.func

        elif isinstance(func, FunctionDef) and func.is_static:
            if parent is None:
                raise ValueError('parent must be given')

        else:
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

    def __new__(cls, module):
        if not isinstance(module, F2PY_Module):
            raise TypeError('Expecting a F2PY_Module')

        return Basic.__new__(cls, module)

    @property
    def module(self):
        return self.args[0]

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

    name = 'f2py_{}'.format(func.name)

    # ...
    results_names = [i.name for i in results]
    _args = []
    _arguments_inout = []
    for i_a,a in enumerate(args):
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
                        is_static = True,
                        arguments_inout = arguments_inout )

