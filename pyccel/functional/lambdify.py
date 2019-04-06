# -*- coding: UTF-8 -*-

# TODO use OrderedDict when possible
#      right now namespace used only globals, => needs to look in locals too

import os
import sys
import importlib
import numpy as np
from types import FunctionType

from sympy import Indexed, IndexedBase, Tuple, Lambda
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import sympify
from sympy import Dummy

from pyccel.codegen.utilities import construct_flags as construct_flags_pyccel
from pyccel.codegen.utilities import execute_pyccel
from pyccel.codegen.utilities import get_source_function
from pyccel.codegen.utilities import random_string
from pyccel.codegen.utilities import write_code
from pyccel.codegen.utilities import mkdir_p
from pyccel.ast.datatypes import dtype_and_precsision_registry as dtype_registry
from pyccel.ast import Variable, Len, Assign, AugAssign
from pyccel.ast import For, Range, FunctionDef
from pyccel.ast import FunctionCall
from pyccel.ast import Comment, AnnotatedComment
from pyccel.ast import Print, Pass, Return
from pyccel.ast.core import Slice, String
from pyccel.ast import Zeros
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.codegen.printing.pycode import pycode
from pyccel.codegen.printing.fcode  import fcode
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.datatypes import get_default_value
from pyccel.functional import Where

from .parser import parse as parse_lambda

#==============================================================================
_accelerator_registery = {'openmp':  'omp',
                          'openacc': 'acc',
                          None:      None}

_known_unary_functions = {'sum': '+',
                          'add': '+',
                          'mul': '*',
                                   }

_known_binary_functions = {}

_known_functions  = dict(_known_unary_functions, **_known_binary_functions)

#==============================================================================
def _lambdify(func, **kwargs):

    if not isinstance(func, FunctionType):
        raise TypeError('Expecting a lambda function')

    # ... get optional arguments
    _kwargs = kwargs.copy()

    namespace = _kwargs.pop('namespace', None)
    # TODO improve using the same way as Equation in sympde
    if namespace is None:
        raise ValueError('namespace must be given')
    # ...

    # ... get the function source code
    func_code = get_source_function(func)
    # ...

    # ...
    ast = parse_lambda(func_code)
    print(func_code)
    print(ast)
    print(ast.name)
    print(ast.body)
    print(ast.body.args)
    print(ast.body.expr)

    import sys; sys.exit(0)
    # ...

    # ...
    calls = list(func.expr.atoms(AppliedUndef))

    typed_functions = {}
    lambdas = {}
    for call in calls:
        # rather than using call.func, we will take the name of the
        # class which defines its type and then the name of the function
        f_name = call.__class__.__name__
        f_symbolic = call.func

        if f_name in namespace.keys():
            f = namespace[f_name]
            decorators = get_decorators(f)
            if f_name in decorators.keys():
                decorators = decorators[f_name]
                if 'types' in decorators:
                    setattr(f_symbolic, '_imp_', f)
                    typed_functions[f_name] = f_symbolic

            else:
                # check if it is a lambda functions
                f_code = get_source_function(f)
                pyccel = Parser(f_code)
                ast = pyccel.parse()
                ls = list(ast.atoms(Lambda))
                if not( len(ls) == 1 ):
                    msg = 'Something wrong happened when parsing {}'.format(f_name)
                    raise ValueError(msg)
                lambdas[f_name] = ls[0]


        # TODO this part is to be removed
        elif (not( f_name in _known_functions.keys() ) and
              not( f_name in  where_stmt.keys()) ):

            raise ValueError('Unkown function {}'.format(f_name))
    print('>>> typed_functions = ', typed_functions)
    print('>>> lambdas         = ', lambdas)
    import sys; sys.exit(0)
    # ...
