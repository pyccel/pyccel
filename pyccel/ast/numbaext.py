#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the numba module understood by pyccel
"""

from functools import reduce
import operator

from pyccel.errors.errors import Errors
from pyccel.errors.messages import WRONG_LINSPACE_ENDPOINT, NON_LITERAL_KEEP_DIMS, NON_LITERAL_AXIS

from pyccel.utilities.stage import PyccelStage

from .basic          import PyccelAstNode
from .builtins       import (PythonInt, PythonBool, PythonFloat, PythonTuple,
                             PythonComplex, PythonReal, PythonImag, PythonList,
                             PythonType, PythonConjugate)

from .core           import Module, Import, PyccelFunctionDef, FunctionCall

from .datatypes      import (dtype_and_precision_registry as dtype_registry,
                             default_precision, datatype, NativeInteger,
                             NativeFloat, NativeComplex, NativeBool, str_dtype,
                             NativeNumeric)

from .internals      import PyccelInternalFunction, Slice, max_precision, get_final_precision
from .internals      import PyccelArraySize

from .literals       import LiteralInteger, LiteralFloat, LiteralComplex, LiteralString, convert_to_literal
from .literals       import LiteralTrue, LiteralFalse
from .literals       import Nil
from .mathext        import MathCeil
from .operators      import broadcast, PyccelMinus, PyccelDiv
from .variable       import (Variable, Constant, HomogeneousTupleVariable)
from .cudaext        import CudaNewArray, CudaArray
from .numpyext       import process_dtype, process_shape, DtypePrecisionToCastFunction

errors = Errors()
pyccel_stage = PyccelStage()

__all__ = (
)

#==============================================================================

numba_funcs = {
    # ... array creation routines
}

numba_constants = {
}


numba_mod = Module('numba',
    variables = numba_constants.values(),
    funcs = numba_funcs.values())

#==============================================================================

numba_target_swap = {

    }
