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
from pyccel.parser import Parser

from .parser import parse    as parse_lambda
from .parser import SemanticParser
from .utilities import get_decorators
from .utilities import get_pyccel_imports_code
from .utilities import get_dependencies_code
from .printing import pycode


#==============================================================================
_accelerator_registery = {'openmp':  'omp',
                          'openacc': 'acc',
                          None:      None}

#==============================================================================
def _parse_typed_functions(user_functions):
    """generate ast for dependencies."""
    code  = get_pyccel_imports_code()
    code += get_dependencies_code(user_functions)

    pyccel = Parser(code)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)
    return ast.namespace.functions

#==============================================================================
def _lambdify(func, namespace={}, **kwargs):

    if not isinstance(func, FunctionType):
        raise TypeError('Expecting a lambda function')

    # ... get the function source code
    func_code = get_source_function(func)
    # ...

    # ...
    L = parse_lambda(func_code)
#    print(L)
#    import sys; sys.exit(0)
    # ...

    # ...
    typed_functions = {}
    for f_name, f in namespace.items():

        # ... check if a typed function
        decorators = get_decorators(f)
        if f_name in decorators.keys():
            decorators = decorators[f_name]
            if 'types' in decorators:
                # TODO
                f_symbolic = f
                typed_functions[f_name] = f_symbolic
                setattr(f_symbolic, '_imp_', f)

            else:
                raise ValueError('{} given without a type'.format(f_name))

        else:
            raise NotImplementedError('')

    typed_functions = _parse_typed_functions(list(typed_functions.values()))
    # ...

    # semantic analysis
    parser = SemanticParser(L, typed_functions=typed_functions)

    # typing
    type_only = kwargs.pop('type_only', False)
    dtype = parser.to_type()

#    print('=========== types ===========')
#    print(parser.d_types)
#    print('=========== expr  ===========')
#    print(parser.d_expr)
#    print('=============================')

    if type_only:
        return dtype

    # annotation
    annotation_only = kwargs.pop('annotation_only', False)
    func = parser.annotate()
    if annotation_only:
        return func

    # printing
    printing_only = kwargs.pop('printing_only', False)
    print(func, type(func))
    code = pycode(func)
    if printing_only:
        return code

    raise NotImplementedError()

    return func
