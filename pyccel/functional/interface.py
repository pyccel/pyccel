# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy import Tuple, IndexedBase
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify
from sympy import FunctionClass


from pyccel.codegen.utilities import random_string
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.core import Slice
from pyccel.ast.core import Variable, FunctionDef, Assign, AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core import FunctionCall
from pyccel.ast.core import Import, Nil, If, Is
from pyccel.ast.core  import For, Range, Len
from pyccel.ast.numpyext  import Zeros
from pyccel.ast.basic import Basic

from .datatypes import TypeVariable, TypeTuple, TypeList
from .semantic import Parser as SemanticParser
from .glossary import _internal_applications
from .glossary import _math_functions
from .glossary import _internal_map_functors
from .ast import BasicGenerator, Shaping, LambdaFunctionDef

#=======================================================================================
def compute_shape( arg, generators ):
    if not( arg in generators.keys() ):
        raise ValueError('Could not find {}'.format( arg ))

    generator = generators[arg]
    return Shaping( generator )


#=======================================================================================
class LambdaInterface(Basic):

    def __new__(cls, func, import_lambda):

        # ...
        m_results = func.m_results

        name    = 'interface_{}'.format(func.name )
        args    = [i for i in func.arguments if not i in m_results]
        s_results = func.results

        results = list(s_results) + list(m_results)
        # ...

        # ...
        imports = [import_lambda]
        stmts   = []
        # ...

        # ... out argument
        if len(results) == 1:
            outs = [Symbol('out')]

        else:
            outs = [Symbol('out_{}'.format(i)) for i in range(0, len(results))]
        # ...

        # ...
        generators = func.generators
        d_shapes = {}
        for i in m_results:
            d_shapes[i] = compute_shape( i, generators )
        # ...

        # ... TODO build statements
        if_cond = Is(Symbol('out'), Nil())

        if_body = []

        # TODO add imports from numpy
        if_body += [Import('zeros', 'numpy')]
        if_body += [Import('float64', 'numpy')]

        for i, var in enumerate(results):
            if var in m_results:
                shaping = d_shapes[var]

                if_body += shaping.stmts
                if_body += [Assign(outs[i], Zeros(shaping.var, var.dtype))]

        # update statements
        stmts = [If((if_cond, if_body))]
        # ...

        # ... add call to the python or pyccelized function
        stmts += [FunctionCall(func, args + outs)]
        # ...

        # ... add return out
        if len(outs) == 1:
            stmts += [Return(outs[0])]

        else:
            stmts += [Return(outs)]
        # ...

        # ...
        body = imports + stmts
        # ...

        # update arguments with optional
        args += [Assign(Symbol('out'), Nil())]

        return FunctionDef( name, args, results, body )

    @property
    def func(self):
        return self.args[0]

    @property
    def out(self):
        return self.args[1]
