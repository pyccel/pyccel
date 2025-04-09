# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import ast
import functools
from itertools import chain, product
import re
import sys
from packaging.version import Version

import numpy as np

from pyccel.ast.basic     import ScopedAstNode

from pyccel.ast.bind_c    import BindCPointer

from pyccel.ast.builtins  import PythonRange, PythonComplex, PythonMin, PythonMax
from pyccel.ast.builtins  import PythonPrint, PythonType, VariableIterator

from pyccel.ast.builtins  import PythonList, PythonTuple, PythonSet, PythonDict, PythonLen

from pyccel.ast.builtin_methods.dict_methods  import DictItems, DictKeys

from pyccel.ast.core      import Declare, For, CodeBlock, ClassDef
from pyccel.ast.core      import FunctionCall, FunctionCallArgument
from pyccel.ast.core      import Deallocate
from pyccel.ast.core      import FunctionAddress
from pyccel.ast.core      import Assign, Import, AugAssign, AliasAssign
from pyccel.ast.core      import SeparatorComment
from pyccel.ast.core      import Module, AsName

from pyccel.ast.c_concepts import ObjectAddress, CMacro, CStringExpression, PointerCast, CNativeInt
from pyccel.ast.c_concepts import CStackArray, CStrStr

from pyccel.ast.datatypes import PythonNativeInt, PythonNativeBool, VoidType
from pyccel.ast.datatypes import TupleType, FixedSizeNumericType, CharType
from pyccel.ast.datatypes import CustomDataType, StringType, HomogeneousTupleType
from pyccel.ast.datatypes import InhomogeneousTupleType, HomogeneousListType, HomogeneousSetType
from pyccel.ast.datatypes import PrimitiveBooleanType, PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveComplexType
from pyccel.ast.datatypes import HomogeneousContainerType, DictType, FixedSizeType

from pyccel.ast.internals import Slice, PrecomputedCode, PyccelArrayShapeElement
from pyccel.ast.internals import PyccelFunction

from pyccel.ast.literals  import LiteralTrue, LiteralFalse, LiteralImaginaryUnit, LiteralFloat
from pyccel.ast.literals  import LiteralString, LiteralInteger, Literal
from pyccel.ast.literals  import Nil, convert_to_literal

from pyccel.ast.low_level_tools import IteratorType

from pyccel.ast.mathext  import math_constants

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpySum
from pyccel.ast.numpyext import NumpyReal, NumpyImag, NumpyFloat
from pyccel.ast.numpyext import NumpyAmin, NumpyAmax
from pyccel.ast.numpyext import get_shape_of_multi_level_container

from pyccel.ast.numpytypes import NumpyInt8Type, NumpyInt16Type, NumpyInt32Type, NumpyInt64Type
from pyccel.ast.numpytypes import NumpyFloat32Type, NumpyFloat64Type, NumpyFloat128Type
from pyccel.ast.numpytypes import NumpyNDArrayType, numpy_precision_map

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelLt, PyccelGt
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelMod
from pyccel.ast.operators import PyccelUnarySub, IfTernaryOperator

from pyccel.ast.type_annotations import VariableTypeAnnotation

from pyccel.ast.utilities import expand_to_loops, is_literal_integer

from pyccel.ast.variable import IndexedElement
from pyccel.ast.variable import Variable
from pyccel.ast.variable import DottedName
from pyccel.ast.variable import DottedVariable

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT, PYCCEL_INTERNAL_ERROR)

numpy_v1 = Version(np.__version__) < Version("2.0.0")

errors = Errors()

# TODO: add examples

__all__ = ["CCodePrinter"]

# dictionary mapping numpy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_NumpyUfuncBase(self, expr)
numpy_ufunc_to_c_float = {
    'NumpyAbs'  : 'fabs',
    'NumpyFabs' : 'fabs',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'log',
    'NumpySqrt': 'sqrt',
    # ---
    'NumpySin'    : 'sin',
    'NumpyCos'    : 'cos',
    'NumpyTan'    : 'tan',
    'NumpyArcsin' : 'asin',
    'NumpyArccos' : 'acos',
    'NumpyArctan' : 'atan',
    'NumpyArctan2': 'atan2',
    'NumpySinh'   : 'sinh',
    'NumpyCosh'   : 'cosh',
    'NumpyTanh'   : 'tanh',
    'NumpyArcsinh': 'asinh',
    'NumpyArccosh': 'acosh',
    'NumpyArctanh': 'atanh',
}

numpy_ufunc_to_c_complex = {
    'NumpyAbs'  : 'cabs',
    # ---
    'NumpyExp' : 'cexp',
    'NumpyLog' : 'clog',
    'NumpySqrt': 'csqrt',
    # ---
    'NumpySin'    : 'csin',
    'NumpyCos'    : 'ccos',
    'NumpyTan'    : 'ctan',
    'NumpyArcsin' : 'casin',
    'NumpyArccos' : 'cacos',
    'NumpyArctan' : 'catan',
    'NumpySinh'   : 'csinh',
    'NumpyCosh'   : 'ccosh',
    'NumpyTanh'   : 'ctanh',
    'NumpyArcsinh': 'casinh',
    'NumpyArccosh': 'cacosh',
    'NumpyArctanh': 'catanh',
}

# dictionary mapping Math function to (argument_conditions, C_function).
# Used in CCodePrinter._print_MathFunctionBase(self, expr)
# Math function ref https://docs.python.org/3/library/math.html
math_function_to_c = {
    # ---------- Number-theoretic and representation functions ------------
    'MathCeil'     : 'ceil',
    # 'MathComb'   : 'com' # TODO
    'MathCopysign': 'copysign',
    'MathFabs'   : 'fabs',
    'MathFloor'    : 'floor',
    # 'MathFmod'   : '???',  # TODO
    # 'MathRexp'   : '???'   TODO requires two output
    # 'MathFsum'   : '???',  # TODO
    # 'MathIsclose' : '???',  # TODO
    'MathIsfinite': 'isfinite', # int isfinite(real-floating x);
    'MathIsinf'   : 'isinf', # int isinf(real-floating x);
    'MathIsnan'   : 'isnan', # int isnan(real-floating x);
    # 'MathIsqrt'  : '???' TODO
    'MathLdexp'  : 'ldexp',
    # 'MathModf'  : '???' TODO return two value
    # 'MathPerm'  : '???' TODO
    # 'MathProd'  : '???' TODO
    'MathRemainder'  : 'remainder',
    'MathTrunc'  : 'trunc',

    # ----------------- Power and logarithmic functions -----------------------

    'MathExp'    : 'exp',
    'MathExpm1'  : 'expm1',
    'MathLog'    : 'log',      # take also an option arg [base]
    'MathLog1p'  : 'log1p',
    'MathLog2'  : 'log2',
    'MathLog10'  : 'log10',
    'MathPow'    : 'pow',
    'MathSqrt'   : 'sqrt',

    # --------------------- Trigonometric functions ---------------------------

    'MathAcos'   : 'acos',
    'MathAsin'   : 'asin',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathCos'    : 'cos',
    # 'MathDist'  : '???', TODO
    'MathHypot'  : 'hypot',
    'MathSin'    : 'sin',
    'MathTan'    : 'tan',


    # -------------------------- Hyperbolic functions -------------------------

    'MathAcosh'  : 'acosh',
    'MathAsinh'  : 'asinh',
    'MathAtanh'  : 'atanh',
    'MathCosh'   : 'cosh',
    'MathSinh'   : 'sinh',
    'MathTanh'   : 'tanh',

    # --------------------------- Special functions ---------------------------

    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathGamma'  : 'tgamma',
    'MathLgamma' : 'lgamma',

    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
    # --------------------------- cmath functions --------------------------
    'CmathAcos'  : 'cacos',
    'CmathAcosh' : 'cacosh',
    'CmathAsin'  : 'casin',
    'CmathAsinh' : 'casinh',
    'CmathAtan'  : 'catan',
    'CmathAtanh' : 'catanh',
    'CmathCos'   : 'ccos',
    'CmathCosh'  : 'ccosh',
    'CmathExp'   : 'cexp',
    'CmathSin'   : 'csin',
    'CmathSinh'  : 'csinh',
    'CmathSqrt'  : 'csqrt',
    'CmathTan'   : 'ctan',
    'CmathTanh'  : 'ctanh',
}

c_library_headers = (
    "complex",
    "ctype",
    "float",
    "inttypes",
    "math",
    "stdarg",
    "stdbool",
    "stddef",
    "stdint",
    "stdio",
    "stdlib",
    "string",
)

import_dict = {'omp_lib' : 'omp' }

c_imports = {n : Import(n, Module(n, (), ())) for n in
                ['assert',
                 'complex',
                 'float',
                 'inttypes',
                 'math',
                 'pyc_math_c',
                 'stdbool',
                 'stdint',
                 'stdio',
                 'stdlib',
                 'string',
                 'stc/cstr',
                 'CSpan_extensions']}

import_header_guard_prefix = {
    'stc/common': '_TOOLS_COMMON',
    'stc/cspan': '', # Included for import sorting
    'stc/hmap': '_TOOLS_DICT',
    'stc/hset': '_TOOLS_SET',
    'stc/vec': '_TOOLS_LIST'
}

stc_extension_mapping = {
    'stc/common': 'STC_Extensions/Common_extensions',
    'stc/hmap': 'STC_Extensions/Dict_extensions',
    'stc/hset': 'STC_Extensions/Set_extensions',
    'stc/vec': 'STC_Extensions/List_extensions',
}

class CCodePrinter(CodePrinter):
    """
    A printer for printing code in C.

    A printer to convert Pyccel's AST to strings of c code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'tabwidth': 4,
    }

    dtype_registry = {CNativeInt()    : 'int',
                      VoidType() : 'void',
                      CharType() : 'char',
                      (PrimitiveComplexType(),8) : 'double complex',
                      (PrimitiveComplexType(),4) : 'float complex',
                      (PrimitiveFloatingPointType(),8)   : 'double',
                      (PrimitiveFloatingPointType(),4)   : 'float',
                      (PrimitiveIntegerType(),4)     : 'int32_t',
                      (PrimitiveIntegerType(),8)     : 'int64_t',
                      (PrimitiveIntegerType(),2)     : 'int16_t',
                      (PrimitiveIntegerType(),1)     : 'int8_t',
                      (PrimitiveBooleanType(),-1) : 'bool',
                      }

    type_to_format = {(PrimitiveFloatingPointType(),8) : '%.15lf',
                      (PrimitiveFloatingPointType(),4) : '%.6f',
                      (PrimitiveIntegerType(),4)       : '%d',
                      (PrimitiveIntegerType(),8)       : LiteralString("%") + CMacro('PRId64'),
                      (PrimitiveIntegerType(),2)       : LiteralString("%") + CMacro('PRId16'),
                      (PrimitiveIntegerType(),1)       : LiteralString("%") + CMacro('PRId8'),
                      }

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename)

        super().__init__()
        self.prefix_module = prefix_module
        self._additional_imports = {'stdlib':c_imports['stdlib']}
        self._additional_code = ''
        self._additional_args = []
        self._temporary_args = []
        self._current_module = None
        self._in_header = False

    def sort_imports(self, imports):
        """
        Sort imports to avoid any errors due to bad ordering.

        Sort imports. This is important so that types exist before they are used to create
        container types. E.g. it is important that complex or inttypes be imported before
        vec_int or vec_double_complex is declared.

        Parameters
        ----------
        imports : list[Import]
            A list of the imports.

        Returns
        -------
        list[Import]
            A sorted list of the imports.
        """
        import_src = [str(i.source) for i in imports]
        dependent_imports = [i for i in import_src if i in import_header_guard_prefix]
        non_stc_imports = [i for i in import_src if i not in dependent_imports]
        dependent_imports.sort()
        non_stc_imports.sort()
        sorted_imports = [imports[import_src.index(name)] for name in chain(non_stc_imports, dependent_imports)]
        return sorted_imports

    def _format_code(self, lines):
        return self.indent_code(lines)

    def is_c_pointer(self, a):
        """
        Indicate whether the object is a pointer in C code.

        Some objects are accessed via a C pointer so that they can be modified in
        their scope and that modification can be retrieved elsewhere. This
        information cannot be found trivially so this function provides that
        information while avoiding easily outdated code to be repeated.

        The main reasons for this treatment are:
        1. It is the actual memory address of an object
        2. It is a reference to another object (e.g. an alias, an optional argument, or one of multiple return arguments)

        See codegen_stage.md in the developer docs for more details.

        Parameters
        ----------
        a : TypedAstNode
            The object whose storage we are enquiring about.

        Returns
        -------
        bool
            True if a C pointer, False otherwise.
        """
        if isinstance(a, (Nil, ObjectAddress, PointerCast, CStrStr)):
            return True
        if isinstance(a, FunctionCall):
            a = a.funcdef.results.var
        # STC _at and _at_mut functions return pointers
        if isinstance(a, IndexedElement) and not isinstance(a.base.class_type, CStackArray) and \
                len(a.indices) == a.base.class_type.container_rank:
            return True
        if not isinstance(a, Variable):
            return False
        if isinstance(a.class_type, (HomogeneousTupleType, NumpyNDArrayType)):
            return a.is_optional or any(a is bi for b in self._additional_args for bi in b)

        if isinstance(a.class_type, (CustomDataType, HomogeneousContainerType, DictType)) \
                and a.is_argument and not a.is_const:
            return True

        return a.is_alias or a.is_optional or \
                any(a is bi for b in self._additional_args for bi in b)

    def _flatten_list(self, irregular_list):
        """
        Get a list of all the arguments in a multi-level list/tuple.

        Get a list of all the arguments in a multi-level list/tuple.

        Parameters
        ----------
        irregular_list : PythonList | PythonTuple
            A multi-level list/tuple.

        Returns
        -------
        list[TypedAstNode]
            A flattened list of all the elements of the list/tuple.
        """
        def to_list(arg):
            """
            Get a list containing the scalar elements of a list/tuple.
            This method is called recursively. If the argument is not
            a list/tuple then it is returned in a list otherwise the
            elements of the list/tuple are converted to a list and
            flattened.
            """
            if isinstance(arg, (PythonList, PythonTuple)):
                return [ai for a in arg.args for ai in to_list(a)]
            else:
                return [arg]
        return to_list(irregular_list)

    #========================== Numpy Elements ===============================#
    def copy_NumpyArray_Data(self, lhs, rhs):
        """
        Get code which copies data from a Ndarray or a homogeneous tuple into a Ndarray.

        When data is copied from a homogeneous tuple, the code declares and fills
        a dummy data_buffer and copies the data from it to a NdArray struct.
        When data is copied from a Ndarray this is done directly without an intermediate
        structure.

        Parameters
        ----------
        lhs : TypedAstNode
            The left-hand side of the assignment containing the array into which the
            NumPy array should be copied.

        rhs : TypedAstNode
            The right-hand side of the assignment containing the array(s) which should
            be copied into the left-hand side.

        Returns
        -------
        str
            A string containing the code which allocates and copies the data.
        """
        assert rhs.rank != 0
        arg = rhs.arg if isinstance(rhs, NumpyArray) else rhs
        lhs_address = self._print(ObjectAddress(lhs))

        variables = [v for v in arg.get_attribute_nodes((Variable, FunctionCall, PyccelFunction)) if v.rank]

        # If the data is copied from a Variable rather than a list or tuple
        # use the function cspan_copy directly
        if isinstance(arg, (Variable, IndexedElement)):
            prefix = ''
            if isinstance(arg, IndexedElement):
                arg_var = self.scope.get_temporary_variable(arg.class_type, memory_handling='heap')
                prefix += self._print(Assign(arg_var, arg))
                arg = arg_var
            if isinstance(arg.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
                rhs_address = self._print(ObjectAddress(arg))
                lhs_c_type = self.get_c_type(lhs.class_type)
                rhs_c_type = self.get_c_type(arg.class_type)
                cast_type = self.get_c_type(lhs.class_type.element_type)
                self.add_import(c_imports['CSpan_extensions'])
                return prefix + f'cspan_copy({cast_type}, {lhs_c_type}, {rhs_c_type}, {lhs_address}, {rhs_address});\n'
            else:
                raise NotImplementedError(f"Can't copy variable of type {arg.class_type}")
        elif variables:
            body = ''
            for li, ri in zip(lhs, arg):
                if li.rank:
                    li_slice_var = self.scope.get_temporary_variable(li.class_type,
                            shape = get_shape_of_multi_level_container(ri), memory_handling='alias')
                    body += self._print(AliasAssign(li_slice_var, li))
                    body += self.copy_NumpyArray_Data(li_slice_var, ri)
                else:
                    body += self._print(Assign(li, ri))
            return body

        def get_indexed(base, elems):
            """
            Get an indexed object. This ensures the necessary levels are created for tuples.
            """
            while elems:
                result = IndexedElement(base, *elems[:base.class_type.container_rank])
                elems = elems[base.class_type.container_rank:]
                base = result
            return result

        flattened_args = self._flatten_list(arg)

        flattened_lhs = [get_indexed(lhs, elems) for elems in product(*[range(s) for s in get_shape_of_multi_level_container(arg)])]

        operations = ''
        for li, ri in zip(flattened_lhs, flattened_args):
            operations += f'{self._print(li)} = {self._print(ri)};\n'

        return operations

    def arrayFill(self, expr):
        """
        Print the assignment of a NdArray.

        Print the code necessary to create and fill an ndarray.

        Parameters
        ----------
        expr : Assign
            The Assign Node used to get the lhs and rhs.

        Returns
        -------
        str
            Return a str that contains a call to the C function array_fill.
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''

        if rhs.fill_value is not None:
            lhs_code = self._print(lhs)
            fill_val = self._print(rhs.fill_value)
            c_type = self.get_c_type(lhs.class_type)
            loop_scope = self.scope.create_new_loop_scope()
            iter_var_name = loop_scope.get_new_name()
            code_init += f'c_foreach({iter_var_name}, {c_type}, {lhs_code}) {{\n'
            code_init += f'*({iter_var_name}.ref) = {fill_val};\n'
            code_init += '}\n'
        return code_init

    def _init_stack_array(self, expr):
        """
        Return a string which handles the assignment of a stack ndarray.

        Print the code necessary to initialise a ndarray on the stack.

        Parameters
        ----------
        expr : TypedAstNode
            The Assign Node used to get the lhs and rhs.

        Returns
        -------
        buffer_array : str
            String initialising the stack (C) array which stores the data.
        array_init   : str
            String containing the rhs of the initialization of a stack array.
        """
        var = expr
        dtype = self.get_c_type(var.dtype)
        shape = ", ".join(self._print(i) for i in var.alloc_shape)
        tot_shape = self._print(functools.reduce(
            lambda x,y: PyccelMul(x,y,simplify=True), var.alloc_shape))

        order = 'c_COLMAJOR' if var.order == 'F' else 'c_ROWMAJOR'

        dummy_array_name = self.scope.get_new_name(f'{var.name}_ptr')
        buffer_array = f"{dtype} {dummy_array_name}[{tot_shape}];\n"
        array_init = f' = cspan_md_layout({order}, {dummy_array_name}, {shape})'
        return buffer_array, array_init

    def _handle_inline_func_call(self, expr):
        """
        Print a function call to an inline function.

        Use the arguments passed to an inline function to print
        its body with the passed arguments in place of the function
        arguments.

        Parameters
        ----------
        expr : FunctionCall
            The function call which should be printed inline.

        Returns
        -------
        str
            The code for the inline function.
        """
        func = expr.funcdef
        body = func.body

        for b in body.body:
            if isinstance(b, ScopedAstNode):
                b.scope.update_parent_scope(self.scope, is_loop=True)

        # Print any arguments using the same inline function
        # As the function definition is modified directly this function
        # cannot be called recursively with the same FunctionDef
        args = []
        for a in expr.args:
            if a.is_user_of(func):
                code = PrecomputedCode(self._print(a))
                args.append(code)
            else:
                args.append(a.value)

        # Create new local variables to ensure there are no name collisions
        new_local_vars = [self.scope.get_temporary_variable(v, clone_scope = func.scope) \
                            for v in func.local_vars]

        parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign))
        func_result_vars = func.scope.collect_all_tuple_elements(func.results.var)
        generated_result_vars = any(v is not Nil() and (not v.is_temp or v.is_ndarray) for v in func_result_vars)
        if generated_result_vars or self._temporary_args:
            if self._temporary_args:
                orig_res_vars = func_result_vars
                new_res_vars = self._temporary_args
            else:
                orig_res_vars = [v for v in func_result_vars if not v.is_temp or v.is_ndarray]
                new_res_vars = [self.scope.get_temporary_variable(r) \
                            for r in orig_res_vars]
            new_res_vars = [a.obj if isinstance(a, ObjectAddress) else a for a in new_res_vars]
            body.substitute(orig_res_vars, new_res_vars)

        # Replace the arguments in the code
        func.swap_in_args(args, new_local_vars)

        func.remove_presence_checks()

        # Collect code but strip empty end
        body_code = self._print(body)
        code_lines = body_code.split('\n')[:-1]
        return_regex = re.compile(r'\breturn\b')
        has_results = [return_regex.search(l) is not None for l in code_lines]

        if func.results.var is Nil() and not any(has_results):
            code = body_code
        else:
            result_idx = has_results.index(True)
            result_line = code_lines[result_idx]

            body_code = '\n'.join(code_lines[:result_idx])+'\n'

            if len(func.results) != 1:
                code = body_code
            else:
                self._additional_code += body_code
                code = result_line.removeprefix('return ').removesuffix(';')

        # Put back original arguments
        func.reinstate_presence_checks()
        func.swap_out_args()
        if generated_result_vars or self._temporary_args:
            body.substitute(new_res_vars, orig_res_vars)

        if func.global_vars or func.global_funcs and \
                not func.get_direct_user_nodes(lambda u: isinstance(u, ClassDef)):
            mod = func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            self.add_import(Import(mod.name, [AsName(v, v.name) \
                for v in (*func.global_vars, *func.global_funcs)]))
            for v in (*func.global_vars, *func.global_funcs):
                self.scope.insert_symbol(v.name)

        for b in body.body:
            if isinstance(b, ScopedAstNode):
                b.scope.update_parent_scope(func.scope, is_loop=True)

        return code

    def init_stc_container(self, expr, assignment_var):
        """
        Generate the initialization of an STC container in C.

        This method generates and prints the C code for initializing a container using the STC `c_init()` method.

        Parameters
        ----------
        expr : TypedAstNode
            The object representing the container being printed (e.g., PythonList, PythonSet).

        assignment_var : Variable
            The variable that the Python container is being assigned to.

        Returns
        -------
        str
            The generated C code for the container initialization.
        """

        class_type = assignment_var.class_type
        dtype = self.get_c_type(class_type)
        if isinstance(expr, PythonDict):
            values = [self._print(k) for k in expr.values]
            if isinstance(class_type.key_type, StringType):
                keys = [self._print(CStrStr(k)) for k in expr.keys]
            else:
                keys = [self._print(k) for k in expr.keys]
            keyraw = '{' + ', '.join(f'{{{k}, {v}}}' for k,v in zip(keys, values)) + '}'
        else:
            if isinstance(class_type.element_type, StringType):
                args = [self._print(CStrStr(a)) for a in expr.args]
            else:
                args = [self._print(a) for a in expr.args]
            keyraw = '{' + ', '.join(args) + '}'
        container_name = self._print(assignment_var)
        init = f'{container_name} = c_init({dtype}, {keyraw});\n'
        return init

    def rename_imported_methods(self, expr):
        """
        Rename class methods from user-defined imports.

        This function is responsible for renaming methods of classes from
        the imported modules, ensuring that the names are correct
        by prefixing them with their class names.

        Parameters
        ----------
        expr : iterable[ClassDef]
            The ClassDef objects found in the module being renamed.
        """
        for classDef in expr:
            class_scope = classDef.scope
            for method in classDef.methods:
                if not method.is_inline:
                    class_scope.rename_function(method, f"{classDef.name}__{method.name.lstrip('__')}")
            for interface in classDef.interfaces:
                for func in interface.functions:
                    if not func.is_inline:
                        class_scope.rename_function(func, f"{classDef.name}__{func.name.lstrip('__')}")

    def _handle_numpy_functional(self, expr, ElementExpression, start_val = None):
        """
        Print code describing a NumPy functional for object.

        Print code describing a NumPy functional for object. E.g. sum/min/max.

        Parameters
        ----------
        expr : TypedAstNode
            The expression to be printed.
        ElementExpression : class type
            A class describing the operation carried out on each element of the
            array argument.
        start_val : TypedAstNode
            The value that the result should be initialised to before the loop.

        Returns
        -------
        str
            Code which describes the NumPy functional calculation.
        """
        assign_node = expr.get_direct_user_nodes(lambda p: isinstance(p, Assign))
        if assign_node:
            lhs_var = assign_node[0].lhs
            arg_var = expr.arg
            class_type = expr.arg.class_type

            prefix = ''

            lhs = self._print(lhs_var)
            if not isinstance(arg_var, Variable):
                # This handles slice arguments
                assert arg_var.rank
                tmp = self.scope.get_temporary_variable(arg_var.class_type, shape = arg_var.shape,
                        memory_handling='alias')
                prefix += self._print(AliasAssign(tmp, arg_var))
                arg_var = tmp
            arg = self._print(arg_var)
            c_type = self.get_c_type(class_type)
            if start_val is None:
                arg_address = self._print(ObjectAddress(arg_var))
                start = f'*cspan_front({arg_address})'
            else:
                start = self._print(start_val)

            loop_scope = self.scope.create_new_loop_scope()
            iter_var_name = loop_scope.get_new_name()
            iter_var = Variable(IteratorType(class_type), iter_var_name)
            iter_ref_var = DottedVariable(arg_var.class_type.element_type, 'ref',
                        memory_handling='alias', lhs=iter_var)

            tmp_additional_code = self._additional_code
            self._additional_code = ''
            node = self._print(ElementExpression(lhs_var, iter_ref_var))
            body = self._additional_code + f'{lhs} = {node};\n'
            self._additional_code = tmp_additional_code

            return prefix + (f'{lhs} = {start};\n'
                    f'c_foreach({iter_var_name}, {c_type}, {arg}) {{\n'
                    f'{body}'
                     '}\n')
        else:
            tmp_var = self.scope.get_temporary_variable(expr.class_type)
            assign_node = Assign(tmp_var, expr)
            self._additional_code += self._handle_numpy_functional(expr,
                            ElementExpression, start_val)
            return self._print(tmp_var)

    # ============ Elements ============ #

    def _print_PythonAbs(self, expr):
        if expr.arg.dtype.primitive_type is PrimitiveFloatingPointType():
            self.add_import(c_imports['math'])
            func = "fabs"
        elif expr.arg.dtype.primitive_type is PrimitiveComplexType():
            self.add_import(c_imports['complex'])
            func = "cabs"
        else:
            func = "labs"
        return "{}({})".format(func, self._print(expr.arg))

    def _print_PythonRound(self, expr):
        self.add_import(c_imports['pyc_math_c'])
        arg = self._print(expr.arg)
        ndigits = self._print(expr.ndigits or LiteralInteger(0))
        if isinstance(expr.arg.class_type.primitive_type, (PrimitiveBooleanType, PrimitiveIntegerType)):
            return f'ipyc_bankers_round({arg}, {ndigits})'
        else:
            return f'fpyc_bankers_round({arg}, {ndigits})'

    def _print_PythonMinMax(self, expr):
        arg = expr.args[0]
        primitive_type = arg.dtype.primitive_type
        variadic_args = isinstance(primitive_type, (PrimitiveFloatingPointType, PrimitiveIntegerType))
        can_compare = primitive_type is not PrimitiveComplexType()

        if isinstance(arg, Variable) and isinstance(arg.class_type , HomogeneousTupleType):
            if isinstance(arg.shape[0], LiteralInteger):
                arg = PythonTuple(*arg)
            else:
                return errors.report(f"{expr.name} in C does not support tuples of unknown length\n"
                                     + PYCCEL_RESTRICTION_TODO, symbol=expr, severity='fatal')
        if isinstance(arg, (PythonTuple, PythonList)) and variadic_args:
            key = self.get_c_type(arg.class_type.element_type)
            self.add_import(Import('stc/common', AsName(VariableTypeAnnotation(arg.dtype), key)))
            args_code = ", ".join(self._print(a) for a in arg.args)
            return  f'{key}_{expr.name}({len(arg.args)}, {args_code})'
        elif isinstance(arg, Variable):
            if isinstance(arg.class_type, (HomogeneousListType, HomogeneousSetType)) and can_compare:
                class_type = arg.class_type
                c_type = self.get_c_type(class_type)
                arg_obj = self._print(ObjectAddress(arg))
                import_loc = 'stc/vec' if isinstance(arg.class_type, HomogeneousListType) else 'stc/hset'
                self.add_import(Import(import_loc, AsName(VariableTypeAnnotation(class_type), c_type)))
                return f'{c_type}_{expr.name}({arg_obj})'
            else:
                return errors.report(f"{expr.name} in C does not support arguments of type {arg.class_type}", symbol=expr,
                    severity='fatal')
        if len(arg) != 2:
            return errors.report(f"{expr.name} in C does not support {len(arg)} arguments of type {arg.dtype}\n"
                                 + PYCCEL_RESTRICTION_TODO, symbol=expr, severity='fatal')
        if primitive_type is PrimitiveFloatingPointType():
            self.add_import(c_imports['math'])
            arg1 = self._print(arg[0])
            arg2 = self._print(arg[1])
            return f"f{expr.name}({arg1}, {arg2})"
        elif isinstance(primitive_type, (PrimitiveBooleanType, PrimitiveIntegerType)):
            if isinstance(arg[0], (Variable, Literal)):
                arg1 = self._print(arg[0])
            else:
                arg1_temp = self.scope.get_temporary_variable(PythonNativeInt())
                assign1 = Assign(arg1_temp, arg[0])
                code = self._print(assign1)
                self._additional_code += code
                arg1 = self._print(arg1_temp)
            if isinstance(arg[1], (Variable, Literal)):
                arg2 = self._print(arg[1])
            else:
                arg2_temp = self.scope.get_temporary_variable(PythonNativeInt())
                assign2 = Assign(arg2_temp, arg[1])
                code = self._print(assign2)
                self._additional_code += code
                arg2 = self._print(arg2_temp)
            op = '<' if isinstance(expr, PythonMin) else '>'
            return f"({arg1} {op} {arg2} ? {arg1} : {arg2})"
        elif isinstance(primitive_type, PrimitiveComplexType):
            self.add_import(c_imports['pyc_math_c'])
            arg1 = self._print(arg[0])
            arg2 = self._print(arg[1])
            return f"complex_{expr.name}({arg1}, {arg2})"
        else:
            return errors.report(f"{expr.name} in C does not support {len(arg)} arguments of type {arg.dtype}\n"
                                 + PYCCEL_RESTRICTION_TODO, symbol=expr, severity='fatal')

    def _print_PythonMin(self, expr):
        return self._print_PythonMinMax(expr)

    def _print_PythonMax(self, expr):
        return self._print_PythonMinMax(expr)

    def _print_SysExit(self, expr):
        code = ""
        if not isinstance(getattr(expr.status.dtype, 'primitive_type', None), PrimitiveIntegerType) \
                or expr.status.rank > 0:
            print_arg = FunctionCallArgument(expr.status)
            code = self._print(PythonPrint((print_arg, ), file="stderr"))
            arg = "1"
        else:
            arg = self._print(expr.status)
        return f"{code}exit({arg});\n"

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        type_name = self.get_c_type(expr.dtype)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonInt(self, expr):
        self.add_import(c_imports['stdint'])
        value = self._print(expr.arg)
        type_name = self.get_c_type(expr.dtype)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonBool(self, expr):
        value = self._print(expr.arg)
        return '({} != 0)'.format(value)

    def _print_Literal(self, expr):
        return repr(expr.python_value)

    def _print_LiteralInteger(self, expr):
        if isinstance(expr, LiteralInteger) and getattr(expr.dtype, 'precision', -1) == 8:
            self.add_import(c_imports['stdint'])
            return f"INT64_C({repr(expr.python_value)})"
        return repr(expr.python_value)

    def _print_LiteralFloat(self, expr):
        if isinstance(expr, LiteralFloat) and expr.dtype.precision == 4:
            return f"{repr(expr.python_value)}f"
        return repr(expr.python_value)

    def _print_LiteralComplex(self, expr):
        if expr.real == LiteralFloat(0):
            return self._print(PyccelAssociativeParenthesis(PyccelMul(expr.imag, LiteralImaginaryUnit())))
        else:
            return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            value = self._print(expr.internal_var)
        else:
            value = self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))
        type_name = self.get_c_type(expr.dtype)
        return '({0})({1})'.format(type_name, value)

    def _print_LiteralImaginaryUnit(self, expr):
        self.add_import(c_imports['complex'])
        return '_Complex_I'

    def _print_Header(self, expr):
        return ''

    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._current_module = expr.module
        self._in_header = True
        name = expr.module.name
        if isinstance(name, AsName):
            name = name.name
        # TODO: Add interfaces
        classes = ""
        funcs = ""
        for classDef in expr.module.classes:
            if classDef.docstring is not None:
                classes += self._print(classDef.docstring)
            classes += f"struct {classDef.name} {{\n"
            # Is external is required to avoid the default initialisation of containers
            attrib_decl = [self._print(Declare(var, external=True)) for var in classDef.attributes]
            classes += ''.join(d.removeprefix('extern ') for d in attrib_decl)
            for method in classDef.methods:
                funcs += f"{self.function_signature(method)};\n"
            for interface in classDef.interfaces:
                for func in interface.functions:
                    funcs += f"{self.function_signature(func)};\n"
            classes += "};\n"
        funcs += '\n'.join(f"{self.function_signature(f)};" for f in expr.module.funcs if not f.is_inline)

        decls = [Declare(v, external=True, module_variable=True) for v in expr.module.variables if not v.is_private]
        global_variables = ''.join(self._print(d) for d in decls)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = self.sort_imports(imports)
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        self._current_module = None
        return (f"#ifndef {name.upper()}_H\n \
                #define {name.upper()}_H\n\n \
                {imports}\n \
                {global_variables}\n \
                {classes}\n \
                {funcs}\n \
                #endif // {name}_H\n")

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr
        for item in expr.imports:
            if item.source_module and item.source_module is not self._current_module:
                self.rename_imported_methods(item.source_module.classes)
        self.rename_imported_methods(expr.classes)
        body    = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join([self._print(d) for d in expr.declarations])

        # Print imports last to be sure that all additional_imports have been collected
        imports = Import(self.scope.get_python_name(expr.name), Module(expr.name,(),()))
        imports = self._print(imports)

        code = ('{imports}\n'
                '{variables}\n'
                '{body}\n').format(
                        imports   = imports,
                        variables = global_variables,
                        body      = body)

        self.exit_scope()
        self._current_module = None
        return code

    def _print_Break(self, expr):
        return 'break;\n'

    def _print_Continue(self, expr):
        return 'continue;\n'

    def _print_While(self, expr):
        self.set_scope(expr.scope)
        body = self._print(expr.body)
        self.exit_scope()
        cond = self._print(expr.test)
        return 'while({condi})\n{{\n{body}}}\n'.format(condi = cond, body = body)

    def _print_If(self, expr):
        lines = []
        condition_setup = []
        for i, (c, b) in enumerate(expr.blocks):
            body = self._print(b)
            if i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else\n")
            else:
                # Print condition
                condition = self._print(c)
                # Retrieve any additional code which cannot be executed in the line containing the condition
                condition_setup.append(self._additional_code)
                self._additional_code = ''
                # Add the condition to the lines of code
                line = f"if ({condition})\n"
                if i == 0:
                    lines.append(line)
                else:
                    lines.append("else " + line)
            lines.append("{\n")
            lines.append(body + "}\n")
        return "".join(chain(condition_setup, lines))

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{cond} ? {true} : {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_LiteralTrue(self, expr):
        return '1'

    def _print_LiteralFalse(self, expr):
        return '0'

    def _print_PyccelAnd(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' && '.join(a for a in args)

    def _print_PyccelOr(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' || '.join(a for a in args)

    def _print_PyccelEq(self, expr):
        lhs, rhs = expr.args
        if isinstance(lhs.class_type, StringType) and isinstance(rhs.class_type, StringType):
            lhs_code = self._print(CStrStr(lhs))
            rhs_code = self._print(CStrStr(rhs))
            return f'!strcmp({lhs_code}, {rhs_code})'
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return f'{lhs_code} == {rhs_code}'

    def _print_PyccelNe(self, expr):
        lhs, rhs = expr.args
        if isinstance(lhs.class_type, StringType) and isinstance(rhs.class_type, StringType):
            lhs_code = self._print(CStrStr(lhs))
            rhs_code = self._print(CStrStr(rhs))
            return f'strcmp({lhs_code}, {rhs_code})'
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return f'{lhs_code} != {rhs_code}'

    def _print_PyccelLt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        return '!{}'.format(a)

    def _print_PyccelIn(self, expr):
        container_type = expr.container.class_type
        element = self._print(expr.element)
        container = self._print(ObjectAddress(expr.container))
        c_type = self.get_c_type(expr.container.class_type)
        if isinstance(container_type, (HomogeneousSetType, DictType)):
            return f'{c_type}_contains({container}, {element})'
        elif isinstance(container_type, HomogeneousListType):
            return f'{c_type}_find({container}, {element}).ref != {c_type}_end({container}).ref'
        else:
            raise errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = expr,
                    severity='fatal')

    def _print_PyccelMod(self, expr):
        self.add_import(c_imports['math'])
        self.add_import(c_imports['pyc_math_c'])

        first = self._print(expr.args[0])
        second = self._print(expr.args[1])

        if expr.dtype.primitive_type is PrimitiveIntegerType():
            return "pyc_modulo({n}, {base})".format(n=first, base=second)

        if expr.args[0].dtype.primitive_type is PrimitiveIntegerType():
            first = self._print(NumpyFloat(expr.args[0]))
        if expr.args[1].dtype.primitive_type is PrimitiveIntegerType():
            second = self._print(NumpyFloat(expr.args[1]))
        return "pyc_fmodulo({n}, {base})".format(n=first, base=second)

    def _print_PyccelPow(self, expr):
        b = expr.args[0]
        e = expr.args[1]

        if expr.dtype.primitive_type is PrimitiveComplexType():
            b = self._print(b if b.dtype.primitive_type is PrimitiveComplexType() else PythonComplex(b))
            e = self._print(e if e.dtype.primitive_type is PrimitiveComplexType() else PythonComplex(e))
            self.add_import(c_imports['complex'])
            return 'cpow({}, {})'.format(b, e)

        self.add_import(c_imports['math'])
        b = self._print(b if b.dtype.primitive_type is PrimitiveFloatingPointType() else NumpyFloat(b))
        e = self._print(e if e.dtype.primitive_type is PrimitiveFloatingPointType() else NumpyFloat(e))
        code = 'pow({}, {})'.format(b, e)
        return self._cast_to(expr, expr.dtype).format(code)

    def _print_Import(self, expr):
        if expr.ignore:
            return ''
        if isinstance(expr.source, AsName):
            source = expr.source.name
        else:
            source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1].python_value
        else:
            source = self._print(source)
        if source == 'stc/cspan':
            native_int_c_type = self.get_c_type(PythonNativeInt())
            code = (f'#define STC_CSPAN_INDEX_TYPE {native_int_c_type}\n'
                    '#include <stc/cspan.h>\n')
            for t in expr.target:
                dtype = t.object.class_type
                container_type = t.local_alias
                element_type = self.get_c_type(dtype.datatype)
                rank = dtype.rank
                header_guard_prefix = import_header_guard_prefix.get(source, '')
                header_guard = f'{header_guard_prefix}_{container_type.upper()}'
                code += ''.join((f'#ifndef {header_guard}\n',
                        f'#define {header_guard}\n',
                        f'using_cspan({container_type}, {element_type}, {rank});\n',
                        f'#endif // {header_guard}\n\n'))
            return code
        elif source == 'stc/common':
            code = ''
            self.add_import(Import(stc_extension_mapping[source], (), ignore_at_print=True))
            for t in expr.target:
                element_decl = f'#define i_key {t.local_alias}\n'
                header_guard_prefix = import_header_guard_prefix.get(source, '')
                header_guard = f'{header_guard_prefix}_{t.local_alias.upper()}'
                code += ''.join((f'#ifndef {header_guard}\n',
                     f'#define {header_guard}\n',
                     element_decl,
                     f'#include <{source}.h>\n',
                     f'#include <{stc_extension_mapping[source]}.h>\n',
                     f'#endif // {header_guard}\n\n'))
            return code
        elif source in import_header_guard_prefix:
            code = ''
            for t in expr.target:
                class_type = t.object.class_type
                container_type = t.local_alias
                self.add_import(Import(stc_extension_mapping[source],
                       AsName(VariableTypeAnnotation(class_type), container_type),
                       ignore_at_print=True))
                if isinstance(class_type, DictType):
                    key_type = class_type.key_type
                    container_key_key = self.get_c_type(class_type.key_type)
                    container_val_key = self.get_c_type(class_type.value_type)
                    container_key = f'{container_key_key}_{container_val_key}'
                    type_decl = f'{container_key_key},{container_val_key}'
                    if isinstance(key_type, FixedSizeType):
                        decl_line = f'#define i_type {container_type},{type_decl}\n'
                    elif isinstance(key_type, StringType):
                        decl_line = (f'#define i_type {container_type}\n'
                                     f'#define i_keypro cstr\n'
                                     f'#define i_val {container_val_key}\n')
                else:
                    element_type = class_type.element_type
                    if isinstance(element_type, FixedSizeType):
                        type_decl = self.get_c_type(element_type)
                        decl_line = f'#define i_type {container_type},{type_decl}\n'
                    elif isinstance(element_type, StringType):
                        decl_line = (f'#define i_type {container_type}\n'
                                     f'#define i_keypro cstr\n')
                    elif isinstance(element_type, (HomogeneousListType, HomogeneousSetType, DictType)):
                        type_decl = self.get_c_type(element_type)
                        decl_line = (f'#define i_type {container_type}\n'
                                     f'#define i_keyclass {type_decl}\n')
                    else:
                        decl_line = ''
                        errors.report(f"The declaration of type {class_type} is not yet implemented.",
                                symbol=expr, severity='error')
                if isinstance(class_type, (HomogeneousListType, HomogeneousSetType)) and isinstance(class_type.element_type, FixedSizeNumericType) \
                        and not isinstance(class_type.element_type.primitive_type, PrimitiveComplexType):
                    decl_line += '#define i_use_cmp\n'
                header_guard_prefix = import_header_guard_prefix.get(source, '')
                header_guard = f'{header_guard_prefix}_{container_type.upper()}'
                code += ''.join((f'#ifndef {header_guard}\n',
                                 f'#define {header_guard}\n',
                                 decl_line,
                                 f'#include <{source}.h>\n'))
                if source in stc_extension_mapping:
                    code += decl_line + f'#include <{stc_extension_mapping[source]}.h>\n'
                code += f'#endif // {header_guard}\n\n'
            return code

        # Get with a default value is not used here as it is
        # slower and on most occasions the import will not be in the
        # dictionary
        if source in import_dict: # pylint: disable=consider-using-get
            source = import_dict[source]


        if source is None:
            return ''
        if expr.source in c_library_headers:
            return '#include <{0}.h>\n'.format(source)
        else:
            return '#include "{0}.h"\n'.format(source)

    def _print_LiteralString(self, expr):
        format_str = format(expr.python_value)
        format_str = format_str.replace("\\", "\\\\")\
                               .replace('\a', '\\a')\
                               .replace('\b', '\\b')\
                               .replace('\f', '\\f')\
                               .replace("\n", "\\n")\
                               .replace('\r', '\\r')\
                               .replace('\t', '\\t')\
                               .replace('\v', '\\v')\
                               .replace('"', '\\"')\
                               .replace("'", "\\'")
        return f'cstr_lit("{format_str}")'

    def get_print_format_and_arg(self, var):
        """
        Get the C print format string for the object var.

        Get the C print format string which will allow the generated code
        to print the variable passed as argument.

        Parameters
        ----------
        var : TypedAstNode
            The object which will be printed.

        Returns
        -------
        arg_format : str
            The format which should be printed in the format string of the
            generated print expression.
        arg : str
            The code which should be printed in the arguments of the generated
            print expression to print the object.
        """
        if isinstance(var.dtype, FixedSizeNumericType):
            primitive_type = var.dtype.primitive_type
            if isinstance(primitive_type, PrimitiveComplexType):
                _, real_part = self.get_print_format_and_arg(NumpyReal(var))
                float_format, imag_part = self.get_print_format_and_arg(NumpyImag(var))
                return f'({float_format} + {float_format}j)', f'{real_part}, {imag_part}'
            elif isinstance(primitive_type, PrimitiveBooleanType):
                return self.get_print_format_and_arg(IfTernaryOperator(var,
                                CStrStr(LiteralString("True")),
                                CStrStr(LiteralString("False"))))
            else:
                try:
                    arg_format = self.type_to_format[(primitive_type, var.dtype.precision)]
                except KeyError:
                    errors.report(f"Printing {var.dtype} type is not supported currently", severity='fatal')
                arg = self._print(var)
        elif isinstance(var.dtype, StringType):
            arg = self._print(CStrStr(var))
            arg_format = '%s'
        elif isinstance(var.dtype, CharType):
            arg = self._print(var)
            arg_format = '%s'
        else:
            try:
                arg_format = self.type_to_format[var.dtype]
            except KeyError:
                errors.report(f"Printing {var.dtype} type is not supported currently", severity='fatal')

            arg = self._print(var)

        return arg_format, arg

    def _print_CStringExpression(self, expr):
        return "".join(self._print(CStrStr(e)) for e in expr.get_flat_expression_list())

    def _print_CMacro(self, expr):
        return str(expr.macro)

    def _print_PythonPrint(self, expr):
        self.add_import(c_imports['stdio'])
        self.add_import(c_imports['inttypes'])
        end = '\n'
        sep = ' '
        code = ''
        empty_end = FunctionCallArgument(LiteralString(''), 'end')
        space_end = FunctionCallArgument(LiteralString(' '), 'end')
        empty_sep = FunctionCallArgument(LiteralString(''), 'sep')
        kwargs = [f for f in expr.expr if f.has_keyword]
        for f in kwargs:
            if f.keyword == 'sep'      :   sep = str(f.value)
            elif f.keyword == 'end'    :   end = str(f.value)
            else: errors.report("{} not implemented as a keyworded argument".format(f.keyword), severity='fatal')
        args_format = []
        args = []
        orig_args = [f for f in expr.expr if not f.has_keyword]

        def formatted_args_to_printf(args_format, args, end):
            args_format = CStringExpression(sep).join(args_format)
            args_format += end
            args_format = self._print(args_format)
            args_code = ', '.join([args_format, *args])
            if expr.file == 'stderr':
                return f"fprintf(stderr, {args_code});\n"
            return f"printf({args_code});\n"

        if len(orig_args) == 0:
            return formatted_args_to_printf(args_format, args, end)

        tuple_start = FunctionCallArgument(LiteralString('('))
        tuple_sep   = LiteralString(', ')
        tuple_end   = FunctionCallArgument(LiteralString(')'))

        for i, f in enumerate(orig_args):
            f = f.value

            if isinstance(f, PythonTuple):
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                args = [FunctionCallArgument(print_arg) for tuple_elem in f for print_arg in (tuple_elem, tuple_sep)][:-1]
                if len(f) == 1:
                    args.append(FunctionCallArgument(LiteralString(',')))
                if i + 1 == len(orig_args):
                    end_of_tuple = FunctionCallArgument(LiteralString(end), 'end')
                else:
                    end_of_tuple = FunctionCallArgument(LiteralString(sep), 'end')
                code += self._print(PythonPrint([tuple_start, *args, tuple_end, empty_sep, end_of_tuple]))
                args = []
                continue
            if isinstance(f, PythonType):
                f = f.print_string

            if isinstance(f, FunctionCall) and isinstance(f.class_type, TupleType):
                tmp_list = [self.scope.get_temporary_variable(a.var.dtype) for a in f.funcdef.results]
                tmp_arg_format_list = []
                for a in tmp_list:
                    arg_format, arg = self.get_print_format_and_arg(a)
                    tmp_arg_format_list.append(arg_format)
                    args.append(arg)
                tmp_arg_format_list = CStringExpression(', ').join(tmp_arg_format_list)
                args_format.append(CStringExpression('(', tmp_arg_format_list, ')'))
                assign = Assign(tmp_list, f)
                code = self._print(assign)
                self._additional_code += code
            elif f.rank > 0 and not isinstance(f.class_type, StringType):
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                for_index = self.scope.get_temporary_variable(PythonNativeInt(), name = 'i')
                max_index = PyccelMinus(f.shape[0], LiteralInteger(1), simplify = True)
                for_range = PythonRange(max_index)
                print_body = [ FunctionCallArgument(f[for_index]) ]
                if f.rank == 1:
                    print_body.append(space_end)

                for_body  = [PythonPrint(print_body, file=expr.file)]
                for_scope = self.scope.create_new_loop_scope()
                for_loop  = For((for_index,), for_range, for_body, scope=for_scope)
                for_end   = FunctionCallArgument(LiteralString(']'+end if i == len(orig_args)-1 else ']'), keyword='end')

                body = CodeBlock([PythonPrint([ FunctionCallArgument(LiteralString('[')), empty_end],
                                                file=expr.file),
                                  for_loop,
                                  PythonPrint([ FunctionCallArgument(f[max_index]), for_end],
                                                file=expr.file)],
                                 unravelled = True)
                code += self._print(body)
            elif isinstance(f, LiteralString):
                args_format.append(f.python_value)
            else:
                arg_format, arg = self.get_print_format_and_arg(f)
                args_format.append(arg_format)
                args.append(arg)
        if args_format:
            code += formatted_args_to_printf(args_format, args, end)
        return code

    def get_c_type(self, dtype):
        """
        Find the corresponding C type of the PyccelType.

        For scalar types, this function searches for the corresponding C data type
        in the `dtype_registry`.  If the provided type is a container (like
        `HomogeneousSetType` or `HomogeneousListType`),  it recursively identifies
        the type of an element of the container and uses it to calculate the
        appropriate type for the `STC` container.
        A `PYCCEL_RESTRICTION_TODO` error is raised if the dtype is not found in the registry.

        Parameters
        ----------
        dtype : PyccelType
            The data type of the expression. This can be a fixed-size numeric type,
            a primitive type, or a container type.

        Returns
        -------
        str
            The code which declares the data type in C or the corresponding `STC` container
            type.

        Raises
        ------
        PyccelCodegenError
            If the dtype is not found in the dtype_registry.
        """
        if isinstance(dtype, FixedSizeNumericType):
            primitive_type = dtype.primitive_type
            if isinstance(primitive_type, PrimitiveComplexType):
                self.add_import(c_imports['complex'])
                return f'{self.get_c_type(dtype.element_type)} complex'
            elif isinstance(primitive_type, PrimitiveIntegerType):
                self.add_import(c_imports['stdint'])
            elif isinstance(dtype, PythonNativeBool):
                self.add_import(c_imports['stdbool'])
                return 'bool'

            key = (primitive_type, dtype.precision)

        elif isinstance(dtype, (NumpyNDArrayType, HomogeneousTupleType)):
            element_type = self.get_c_type(dtype.datatype).replace(' ', '_').rstrip('_t')
            i_type = f'array_{element_type}_{dtype.rank}d'
            self.add_import(Import('stc/cspan', AsName(VariableTypeAnnotation(dtype), i_type)))
            return i_type

        elif isinstance(dtype, (HomogeneousSetType, HomogeneousListType)):
            container_type = 'hset' if dtype.name == 'set' else 'vec'
            element_type = self.get_c_type(dtype.element_type).replace(' ', '_')
            i_type = f'{container_type}_{element_type}'
            self.add_import(Import(f'stc/{container_type}', AsName(VariableTypeAnnotation(dtype), i_type)))
            return i_type

        elif isinstance(dtype, DictType):
            container_type = 'hmap'
            key_type = self.get_c_type(dtype.key_type).replace(' ', '_')
            val_type = self.get_c_type(dtype.value_type).replace(' ', '_')
            i_type = f'{container_type}_{key_type}_{val_type}'
            self.add_import(Import(f'stc/{container_type}', AsName(VariableTypeAnnotation(dtype), i_type)))
            return i_type

        elif isinstance(dtype, StringType):
            self.add_import(c_imports['stc/cstr'])
            return 'cstr'
        else:
            key = dtype

        try :
            return self.dtype_registry[key]
        except KeyError:
            raise errors.report(PYCCEL_RESTRICTION_TODO, #pylint: disable=raise-missing-from
                    symbol = dtype,
                    severity='fatal')

    def get_declare_type(self, expr):
        """
        Get the string which describes the type in a declaration.

        This function returns the code which describes the type
        of the `expr` object such that the declaration can be written as:
        `f"{self.get_declare_type(expr)} {expr.name}"`
        The function takes care of reporting errors for unknown types and
        importing any necessary additional imports (e.g. stdint/ndarrays).

        Parameters
        ----------
        expr : Variable
            The variable whose type should be described.

        Returns
        -------
        str
            The code describing the type.

        Raises
        ------
        PyccelCodegenError
            If the type is not supported in the C code.

        Examples
        --------
        >>> v = Variable(PythonNativeInt(), 'x')
        >>> self.get_declare_type(v)
        'int64_t'

        For an object accessed via a pointer:
        >>> v = Variable(NumpyNDArrayType(PythonNativeInt(), 1, None), 'x', is_optional=True)
        >>> self.get_declare_type(v)
        'array_int64_1d*'
        """
        class_type = expr.class_type

        if isinstance(expr.class_type, CStackArray):
            dtype = self.get_c_type(expr.class_type.element_type)
        elif isinstance(expr.class_type, (HomogeneousContainerType, DictType)):
            dtype = self.get_c_type(expr.class_type)
        elif not isinstance(class_type, CustomDataType):
            dtype = self.get_c_type(expr.dtype)
        else:
            dtype = self._print(expr.class_type)

        if getattr(expr, 'is_const', False):
            dtype = f'const {dtype}'

        if self.is_c_pointer(expr) and not isinstance(expr.class_type, CStackArray):
            return f'{dtype}*'
        else:
            return dtype

    def _print_Declare(self, expr):
        var = expr.variable
        if isinstance(var.class_type, InhomogeneousTupleType):
            return ''

        declaration_type = self.get_declare_type(var)

        init = f' = {self._print(expr.value)}' if expr.value is not None else ''

        if isinstance(var.class_type, CStackArray):
            assert init == ''
            preface = ''
            if isinstance(var.alloc_shape[0], (int, LiteralInteger)):
                init = f'[{var.alloc_shape[0]}]'
            else:
                declaration_type += '*'
                init = ''
        elif var.is_stack_array:
            preface, init = self._init_stack_array(var)
        else:
            preface = ''
            if isinstance(var.class_type, (HomogeneousContainerType, DictType)) and not expr.external:
                init = ' = {0}'

        external = 'extern ' if expr.external else ''
        static = 'static ' if expr.static else ''

        return f'{preface}{static}{external}{declaration_type} {var.name}{init};\n'

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the C representation of the function signature.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature and return a string describing the
        function.

        This is not a declaration as the signature does not end
        with a semi-colon.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        print_arg_names : bool, default : True
            Indicates whether argument names should be printed.

        Returns
        -------
        str
            Signature of the function.
        """
        arg_vars = [a.var for a in expr.arguments]
        arg_vars = [ai for a in arg_vars for ai in expr.scope.collect_all_tuple_elements(a)]
        result_vars = [v for v in expr.scope.collect_all_tuple_elements(expr.results.var) \
                            if v and not v.is_argument]

        n_results = len(result_vars)

        if n_results > 1 or isinstance(expr.results.var.class_type, InhomogeneousTupleType):
            ret_type = self.get_c_type(PythonNativeInt())
            arg_vars.extend(result_vars)
            self._additional_args.append(result_vars) # Ensure correct result for is_c_pointer
        elif n_results == 1:
            ret_type = self.get_declare_type(result_vars[0])
        else:
            ret_type = self.get_c_type(VoidType())

        name = expr.name
        if not arg_vars:
            arg_code = 'void'
        else:
            def get_arg_declaration(var):
                """ Get the code which declares the argument variable.
                """
                code = self.get_declare_type(var)
                if print_arg_names:
                    code += ' ' + var.name
                return code

            arg_code_list = [self.function_signature(var, False) if isinstance(var, FunctionAddress)
                                else get_arg_declaration(var) for var in arg_vars]
            arg_code = ', '.join(arg_code_list)

        if self._additional_args :
            self._additional_args.pop()

        static = 'static ' if expr.is_static else ''

        if isinstance(expr, FunctionAddress):
            return f'{static}{ret_type} (*{name})({arg_code})'
        else:
            return f'{static}{ret_type} {name}({arg_code})'

    def _print_IndexedElement(self, expr):
        base = expr.base
        inds = list(expr.indices)
        base_shape = base.shape
        allow_negative_indexes = expr.allows_negative_indexes

        if isinstance(base, IndexedElement):
            while isinstance(base, IndexedElement) and isinstance(base.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
                inds = list(base.indices) + inds
                base = base.base

        if expr.rank > 0 and isinstance(base.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            c_type = self.get_c_type(expr.class_type)
            indices = []
            for i,idx in enumerate(inds):
                if isinstance(idx, Slice):
                    idx = self._new_slice_with_processed_arguments(idx, PyccelArrayShapeElement(base, i),
                        allow_negative_indexes)
                indices.append('{'+self._print(idx)+'}')
            indices_code = ', '.join(indices)
            base_code = self._print(ObjectAddress(base))
            return f'cspan_slice({c_type}, {base_code}, {indices_code})'

        for i, ind in enumerate(inds):
            if is_literal_integer(ind) and int(ind) < 0:
                inds[i] = PyccelMinus(base_shape[i], ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind, simplify = True), ind)

        if isinstance(base.class_type, HomogeneousListType):
            assign = expr.get_user_nodes(Assign)
            index = self._print(inds[0])
            list_var = self._print(ObjectAddress(base))
            container_type = self.get_c_type(base.class_type)
            if assign:
                is_mut = False
                for a in assign:
                    lhs = a.lhs
                    is_mut = is_mut or lhs == expr or lhs.is_user_of(expr)
                if is_mut:
                    return f"(*{container_type}_at_mut({list_var},{index}))"
            return f"(*{container_type}_at({list_var},{index}))"

        indices = ", ".join(self._print(i) for i in inds)
        if isinstance(base.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            return f'(*cspan_at({self._print(ObjectAddress(base))}, {indices}))'
        elif isinstance(base.class_type, CStackArray):
            return f'{self._print(ObjectAddress(base))}[{indices}]'
        else:
            raise NotImplementedError(f"Indexing not implemented for {base.class_type}")

    def _cast_to(self, expr, dtype):
        """
        Add a cast to an expression when needed.

        Get a format string which provides the code to cast the object `expr`
        to the specified dtype. If the dtypes already
        match then the format string will simply print the expression.

        Parameters
        ----------
        expr : TypedAstNode
            The expression to be cast.
        dtype : PyccelType
            The target type of the cast.

        Returns
        -------
        str
            A format string that contains the desired cast type.
            NB: You should insert the expression to be cast in the string
            after using this function.
        """
        if expr.dtype != dtype:
            cast=self.get_c_type(dtype)
            return '({}){{}}'.format(cast)
        return '{}'

    def _print_DottedVariable(self, expr):
        """convert dotted Variable to their C equivalent"""

        name_code = self._print(expr.name)
        if self.is_c_pointer(expr.lhs):
            code = f'{self._print(ObjectAddress(expr.lhs))}->{name_code}'
        else:
            lhs_code = self._print(expr.lhs)
            code = f'{lhs_code}.{name_code}'
        if self.is_c_pointer(expr):
            return f'(*{code})'
        else:
            return code

    @staticmethod
    def _new_slice_with_processed_arguments(_slice, array_size, allow_negative_index):
        """
        Create new slice with information collected from old slice and decorators.

        Create a new slice where the original `start`, `stop`, and `step` have
        been processed using basic simplifications, as well as additional rules
        identified by the function decorators.

        Parameters
        ----------
        _slice : Slice
            Slice needed to collect (start, stop, step).

        array_size : PyccelArrayShapeElement
            Call to function size().

        allow_negative_index : bool
            True when the decorator allow_negative_index is present.

        Returns
        -------
        Slice
            The new slice with processed arguments (start, stop, step).
        """
        start = LiteralInteger(0) if _slice.start is None else _slice.start
        stop = array_size if _slice.stop is None else _slice.stop

        # negative start and end in slice
        if isinstance(start, PyccelUnarySub) and isinstance(start.args[0], LiteralInteger):
            start = PyccelMinus(array_size, start.args[0], simplify = True)
        elif allow_negative_index and not isinstance(start, (LiteralInteger, PyccelArrayShapeElement)):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                            PyccelMinus(array_size, start, simplify = True), start)
        else:
            start = _slice.start

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0], simplify = True)
        elif allow_negative_index and not isinstance(stop, (LiteralInteger, PyccelArrayShapeElement)):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                            PyccelMinus(array_size, stop, simplify = True), stop)
        else:
            stop = _slice.stop

        # steps in slices
        step = _slice.step

        # negative step in slice
        if step and isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            start = PyccelMinus(array_size, LiteralInteger(1), simplify = True) if _slice.start is None else start
            stop = LiteralInteger(0) if _slice.stop is None else stop
            raise NotImplementedError("Negative step not yet handled")

        # variable step in slice
        elif step and allow_negative_index and step and not isinstance(step, LiteralInteger):
            og_start = start
            start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), start, PyccelMinus(stop, LiteralInteger(1), simplify = True))
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), stop, og_start)
            raise NotImplementedError("Negative step not yet handled")

        return Slice(start, stop, step)

    def _print_PyccelArraySize(self, expr):
        arg = self._print(ObjectAddress(expr.arg))
        return f'cspan_size({arg})'

    def _print_PyccelArrayShapeElement(self, expr):
        arg = expr.arg
        if isinstance(arg.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            idx = self._print(expr.index)
            if self.is_c_pointer(arg):
                arg_code = self._print(ObjectAddress(arg))
                return f'{arg_code}->shape[{idx}]'
            arg_code = self._print(arg)
            return f'{arg_code}.shape[{idx}]'
        elif isinstance(arg.class_type, (HomogeneousListType, HomogeneousSetType, DictType)):
            c_type = self.get_c_type(arg.class_type)
            arg_code = self._print(ObjectAddress(arg))
            return f'{c_type}_size({arg_code})'
        elif isinstance(arg.class_type, StringType):
            arg_code = self._print(ObjectAddress(arg))
            return f'cstr_size({arg_code})'
        else:
            raise NotImplementedError(f"Don't know how to represent shape of object of type {arg.class_type}")

    def _print_Allocate(self, expr):
        free_code = ''
        variable = expr.variable
        if isinstance(variable.class_type, (HomogeneousListType, HomogeneousSetType, DictType, StringType)):
            if expr.status in ('allocated', 'unknown'):
                free_code = f'{self._print(Deallocate(variable))}'
            if expr.shape[0] is None:
                return free_code
            size = self._print(expr.shape[0])
            variable_address = self._print(ObjectAddress(expr.variable))
            container_type = self.get_c_type(expr.variable.class_type)
            if expr.alloc_type == 'reserve':
                if expr.status != 'unallocated':
                    return (f'{container_type}_clear({variable_address});\n'
                            f'{container_type}_reserve({variable_address}, {size});\n')
                return f'{container_type}_reserve({variable_address}, {size});\n'
            elif expr.alloc_type == 'resize':
                return f'{container_type}_resize({variable_address}, {size}, {0});\n'
            return free_code
        elif isinstance(variable.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            #free the array if its already allocated and checking if its not null if the status is unknown
            if  (expr.status == 'unknown'):
                data_ptr = ObjectAddress(DottedVariable(VoidType(), 'data',
                                        lhs = variable, memory_handling='alias'))
                free_code = f'if ({self._print(data_ptr)} != NULL)\n'
                free_code += ''.join(("{\n", self._print(Deallocate(variable)), "}\n"))
            elif (expr.status == 'allocated'):
                free_code += self._print(Deallocate(variable))

            tot_shape = self._print(functools.reduce(
                lambda x,y: PyccelMul(x,y,simplify=True), expr.shape))
            c_type = self.get_c_type(variable.class_type)
            element_type = self.get_c_type(variable.class_type.element_type)

            if expr.like:
                buffer_array = ''
                if isinstance(expr.like.class_type, VoidType):
                    dummy_array_name = self._print(ObjectAddress(expr.like))
                else:
                    raise NotImplementedError("Unexpected type passed to like argument")
            else:
                dummy_array_name = self.scope.get_new_name(f'{variable.name}_ptr')
                buffer_array_var = Variable(variable.class_type.datatype, dummy_array_name, memory_handling='alias')
                self.scope.insert_variable(buffer_array_var)
                buffer_array = f"{dummy_array_name} = malloc(sizeof({element_type}) * ({tot_shape}));\n"

            order = 'c_COLMAJOR' if variable.order == 'F' else 'c_ROWMAJOR'
            shape = ", ".join(self._print(i) for i in expr.shape)

            return free_code + buffer_array + f'{self._print(variable)} = ({c_type})cspan_md_layout({order}, {dummy_array_name}, {shape});\n'
        elif variable.is_alias:
            var_code = self._print(ObjectAddress(variable))
            if expr.like:
                declaration_type = self.get_declare_type(expr.like)
                malloc_size = f'sizeof({declaration_type})'
                if variable.rank:
                    tot_shape = self._print(functools.reduce(
                        lambda x,y: PyccelMul(x,y,simplify=True), expr.shape))
                    malloc_size = f'{malloc_size} * ({tot_shape})'
                return f'{var_code} = malloc({malloc_size});\n'
            else:
                raise NotImplementedError(f"Allocate not implemented for {variable.class_type}")
        else:
            raise NotImplementedError(f"Allocate not implemented for {variable.class_type}")

    def _print_Deallocate(self, expr):
        if isinstance(expr.variable.class_type, (HomogeneousListType, HomogeneousSetType, DictType, StringType)):
            if expr.variable.is_alias:
                return ''
            variable_address = self._print(ObjectAddress(expr.variable))
            container_type = self.get_c_type(expr.variable.class_type)
            return f'{container_type}_drop({variable_address});\n'
        if isinstance(expr.variable.class_type, InhomogeneousTupleType):
            return ''.join(self._print(Deallocate(v)) for v in expr.variable)
        if isinstance(expr.variable.dtype, CustomDataType):
            variable_address = self._print(ObjectAddress(expr.variable))
            Pyccel__del = expr.variable.cls_base.scope.find('__del__').name
            return f"{Pyccel__del}({variable_address});\n"
        elif isinstance(expr.variable.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            if expr.variable.is_alias:
                return ''
            else:
                data_ptr = DottedVariable(VoidType(), 'data', lhs = expr.variable, memory_handling='alias')
                data_ptr_code = self._print(ObjectAddress(data_ptr))
                return f'free({data_ptr_code});\n{data_ptr_code} = NULL;\n'
        else:
            variable_address = self._print(ObjectAddress(expr.variable))
            return f'free({variable_address});\n'

    def _print_Slice(self, expr):
        start = expr.start
        stop = expr.stop
        step = expr.step
        if start is None and stop is None and (step in (None, 1)):
            return 'c_ALL'
        else:
            start = start or LiteralInteger(0)
            stop = stop or 'c_END'
            assert not(is_literal_integer(start) and int(start) < 0)
            assert not(is_literal_integer(stop) and int(stop) < 0)
            args = (start, stop)
            if step and step != 1:
                args += (expr.step, )
            return ', '.join(self._print(a) for a in args)

    def _print_NumpyUfuncBase(self, expr):
        """ Convert a Python expression with a Numpy function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Numpy function call

        Returns
        -------
            string
                Equivalent expression in C language

        Example
        -------
            numpy.cos(x) ==> cos(x)

        """
        # add necessary include
        self.add_import(c_imports['math'])
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_c_float[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype.primitive_type is PrimitiveComplexType():
                self.add_import(c_imports['complex'])
                try:
                    func_name = numpy_ufunc_to_c_complex[type_name]
                    args.append(self._print(arg))
                except KeyError:
                    errors.report(INCOMPATIBLE_TYPEVAR_TO_FUNC.format(type_name) ,severity='fatal')
            elif arg.dtype.primitive_type is not PrimitiveFloatingPointType():
                args.append(self._print(NumpyFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_NumpySign(self, expr):
        """ Print the corresponding C function for a call to Numpy.sign

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with Numpy.sign call

        Returns
        -------
            string
                Equivalent internal function in C

        Example
        -------
            import numpy

            numpy.sign(x) => isign(x)   (x is integer)
            numpy.sign(x) => fsign(x)   (x if float)
            numpy.sign(x) => csign(x)   (x is complex)

        """
        self.add_import(c_imports['pyc_math_c'])
        primitive_type = expr.dtype.primitive_type
        func = ''
        if isinstance(primitive_type, PrimitiveIntegerType):
            func = 'isign'
        elif isinstance(primitive_type, PrimitiveFloatingPointType):
            func = 'fsign'
        elif isinstance(primitive_type, PrimitiveComplexType):
            func = 'csgn' if numpy_v1 else 'csign'

        return f'{func}({self._print(expr.args[0])})'

    def _print_NumpyIsFinite(self, expr):
        """
        Convert a Python expression with a numpy isfinite function call to C function call
        """

        self.add_import(c_imports['math'])
        code_arg = self._print(expr.arg)
        return f"isfinite({code_arg})"

    def _print_NumpyIsInf(self, expr):
        """
        Convert a Python expression with a numpy isinf function call to C function call
        """

        self.add_import(c_imports['math'])
        code_arg = self._print(expr.arg)
        return f"isinf({code_arg})"

    def _print_NumpyIsNan(self, expr):
        """
        Convert a Python expression with a numpy isnan function call to C function call
        """

        self.add_import(c_imports['math'])
        code_arg = self._print(expr.arg)
        return f"isnan({code_arg})"

    def _print_MathFunctionBase(self, expr):
        """ Convert a Python expression with a math function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math function call

        Returns
        -------
            string
                Equivalent expression in C language

        ------
        Example:
        --------
            math.sin(x) ==> sin(x)

        """
        # add necessary include
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_c[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        if func_name.startswith("pyc"):
            self.add_import(c_imports['pyc_math_c'])
        else:
            if expr.dtype.primitive_type is PrimitiveComplexType():
                self.add_import(c_imports['complex'])
            else:
                self.add_import(c_imports['math'])
        if expr.dtype.primitive_type is PrimitiveComplexType():
            args = [self._print(a) for a in expr.args]
        else:
            args = []
            for arg in expr.args:
                if arg.dtype.primitive_type is not PrimitiveFloatingPointType() \
                        and not func_name.startswith("pyc"):
                    args.append(self._print(NumpyFloat(arg)))
                else:
                    args.append(self._print(arg))
        code_args = ', '.join(args)
        if expr.dtype.primitive_type is PrimitiveIntegerType():
            cast_type = self.get_c_type(expr.dtype)
            return f'({cast_type}){func_name}({code_args})'
        return f'{func_name}({code_args})'

    def _print_MathIsfinite(self, expr):
        """Convert a Python expression with a math isfinite function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype.primitive_type is PrimitiveIntegerType():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isfinite({})".format(code_arg)

    def _print_MathIsinf(self, expr):
        """Convert a Python expression with a math isinf function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype.primitive_type is PrimitiveIntegerType():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isinf({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype.primitive_type is PrimitiveIntegerType():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype.primitive_type is PrimitiveIntegerType():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "trunc({})".format(code_arg)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_NumpyWhere(self, expr):
        cond = self._print(expr.condition)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        stmt = '{cond} ? {true} : {false}'.format(cond = cond,
                true = value_true, false = value_false)
        return stmt

    def _print_Rand(self, expr):
        raise NotImplementedError("Rand not implemented")

    def _print_NumpyRandint(self, expr):
        raise NotImplementedError("Randint not implemented")

    def _print_NumpyMod(self, expr):
        return self._print(PyccelMod(*expr.args))

    def _print_NumpyAmax(self, expr):
        '''
        Convert a call to numpy.max to the equivalent function in C.
        '''
        return self._handle_numpy_functional(expr, PythonMax)

    def _print_NumpyAmin(self, expr):
        '''
        Convert a call to numpy.min to the equivalent function in C.
        '''
        return self._handle_numpy_functional(expr, PythonMin)

    def _print_NumpySum(self, expr):
        return self._handle_numpy_functional(expr, PyccelAdd, convert_to_literal(0, expr.class_type))

    def _print_NumpyLinspace(self, expr):
        template = '({start} + {index}*{step})'
        if not isinstance(expr.endpoint, LiteralFalse):
            template = '({start} + {index}*{step})'
            lhs_source = expr.get_user_nodes(Assign)[0].lhs
            lhs_source.substitute(expr.ind, PyccelMinus(expr.num, LiteralInteger(1), simplify = True))
            lhs = self._print(lhs_source)

            if isinstance(expr.endpoint, LiteralTrue):
                cond_template = lhs + ' = {stop}'
            else:
                cond_template = lhs + ' = {cond} ? {stop} : ' + lhs

        v = self._cast_to(expr.stop, expr.dtype).format(self._print(expr.stop))

        init_value = template.format(
            start = self._print(expr.start),
            step  = self._print(expr.step),
            index = self._print(expr.ind),
        )
        if isinstance(expr.endpoint, LiteralFalse):
            code = init_value
        elif isinstance(expr.endpoint, LiteralTrue):
            code = init_value + ';\n' + cond_template.format(stop = v)
        else:
            code = init_value + ';\n' + cond_template.format(cond=self._print(expr.endpoint),stop = v)

        return code

    def _print_Interface(self, expr):
        return ""

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''

        self.set_scope(expr.scope)

        arguments = [a.var for a in expr.arguments]
        # Collect results filtering out Nil()
        results = [r for r in self.scope.collect_all_tuple_elements(expr.results.var) \
                if isinstance(r, Variable)]
        returning_tuple = isinstance(expr.results.var.class_type, InhomogeneousTupleType)
        if len(results) > 1 or returning_tuple:
            self._additional_args.append(results)

        body  = self._print(expr.body)
        decs = [Declare(i, value=(Nil() if i.is_alias and isinstance(i.class_type, (VoidType, BindCPointer)) else None))
                for i in expr.local_vars]

        if len(results) == 1 and not returning_tuple:
            res = results[0]
            if isinstance(res, Variable) and not res.is_temp:
                decs += [Declare(res)]
            elif not isinstance(res, Variable):
                raise NotImplementedError(f"Can't return {type(res)} from a function")
        decs  = ''.join(self._print(i) for i in decs)

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        for i in expr.imports:
            self.add_import(i)
        docstring = self._print(expr.docstring) if expr.docstring else ''

        parts = [sep,
                 docstring,
                '{signature}\n{{\n'.format(signature=self.function_signature(expr)),
                 decs,
                 body,
                 '}\n',
                 sep]

        self.exit_scope()

        return ''.join(p for p in parts if p)

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        if func.is_inline:
            return self._handle_inline_func_call(expr)
         # Ensure the correct syntax is used for pointers
        args = []
        for a, f in zip(expr.args, func.arguments):
            arg_val = a.value or Nil()
            f = f.var
            if self.is_c_pointer(f):
                if isinstance(arg_val, Variable):
                    args.append(ObjectAddress(arg_val))
                elif not self.is_c_pointer(arg_val):
                    tmp_var = self.scope.get_temporary_variable(f.dtype)
                    assign = Assign(tmp_var, arg_val)
                    code = self._print(assign)
                    self._additional_code += code
                    args.append(ObjectAddress(tmp_var))
                else:
                    args.append(arg_val)
            else :
                args.append(arg_val)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join(self._print(ai) for a in args for ai in self.scope.collect_all_tuple_elements(a))

        call_code = f'{func.name}({args})'
        if func.results.var is not Nil() and \
                not isinstance(func.results.var.class_type, InhomogeneousTupleType):
            return call_code
        else:
            return f'{call_code};\n'

    def _print_Return(self, expr):
        code = ''
        return_obj = expr.expr
        if return_obj is None:
            args = []
        elif isinstance(return_obj.class_type, InhomogeneousTupleType):
            if expr.stmt:
                return self._print(expr.stmt)+'return 0;\n'
            return 'return 0;\n'
        else:
            args = [ObjectAddress(return_obj) if self.is_c_pointer(return_obj) else return_obj]

        if len(args) == 0:
            return 'return;\n'

        if len(args) > 1:
            if expr.stmt:
                return self._print(expr.stmt)+'return 0;\n'
            return 'return 0;\n'

        if expr.stmt:
            # get Assign nodes from the CodeBlock object expr.stmt.
            last_assign = expr.stmt.get_attribute_nodes((Assign, AliasAssign), excluded_nodes=FunctionCall)
            deallocate_nodes = expr.stmt.get_attribute_nodes(Deallocate, excluded_nodes=(Assign,))
            vars_in_deallocate_nodes = [i.variable for i in deallocate_nodes]

            # Check the Assign objects list in case of
            # the user assigns a variable to an object contains IndexedElement object.
            if not last_assign:
                code = self._print(expr.stmt)
            elif isinstance(last_assign[-1], (AugAssign, AliasAssign)):
                last_assign[-1].lhs.is_temp = False
                code = self._print(expr.stmt)
            elif isinstance(last_assign[-1].rhs, PythonTuple):
                code = self._print(expr.stmt)
                last_assign[-1].lhs.is_temp = False
            else:
                # make sure that stmt contains one assign node.
                last_assign = last_assign[-1]
                variables = last_assign.rhs.get_attribute_nodes(Variable)
                unneeded_var = not any(b in vars_in_deallocate_nodes or b.is_ndarray for b in variables) and \
                        isinstance(last_assign.lhs, Variable) and not last_assign.lhs.is_ndarray
                if unneeded_var:
                    code = ''.join(self._print(a) for a in expr.stmt.body if a is not last_assign)
                    return code + 'return {};\n'.format(self._print(last_assign.rhs))
                else:
                    if isinstance(last_assign.lhs, Variable):
                        last_assign.lhs.is_temp = False
                    code = self._print(expr.stmt)

        returned_value = self.scope.collect_tuple_element(args[0])

        return code + f'return {self._print(returned_value)};\n'

    def _print_Pass(self, expr):
        return '// pass\n'

    def _print_Nil(self, expr):
        return 'NULL'

    def _print_NilArgument(self, expr):
        raise errors.report("Trying to use optional argument in inline function without providing a variable",
                symbol=expr,
                severity='fatal')

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype.primitive_type is PrimitiveIntegerType() for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return  ' / '.join(self._print(a) for a in args)

    def _print_PyccelFloorDiv(self, expr):
        # the result type of the floor division is dependent on the arguments
        # type, if all arguments are integers or booleans the result is integer
        # otherwise the result type is float
        need_to_cast = all(a.dtype.primitive_type in (PrimitiveIntegerType(), PrimitiveBooleanType()) for a in expr.args)
        if need_to_cast:
            self.add_import(c_imports['pyc_math_c'])
            cast_type = self.get_c_type(expr.dtype)
            return f'py_floor_div_{cast_type}({self._print(expr.args[0])}, {self._print(expr.args[1])})'

        self.add_import(c_imports['math'])
        code = ' / '.join(self._print(a if a.dtype.primitive_type is PrimitiveFloatingPointType()
                                        else NumpyFloat(a)) for a in expr.args)
        return f"floor({code})"

    def _print_PyccelRShift(self, expr):
        return ' >> '.join(self._print(a) for a in expr.args)

    def _print_PyccelLShift(self, expr):
        return ' << '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitXor(self, expr):
        if expr.dtype is PythonNativeBool():
            return '{0} != {1}'.format(self._print(expr.args[0]), self._print(expr.args[1]))
        return ' ^ '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is PythonNativeBool():
            return ' || '.join(self._print(a) for a in expr.args)
        return ' | '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is PythonNativeBool():
            return ' && '.join(self._print(a) for a in expr.args)
        return ' & '.join(self._print(a) for a in expr.args)

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_AugAssign(self, expr):
        op = expr.op
        lhs = expr.lhs
        rhs = expr.rhs

        if op == '%' and isinstance(lhs.dtype.primitive_type, PrimitiveFloatingPointType):
            _expr = expr.to_basic_assign()
            expr.invalidate_node()
            return self._print(_expr)

        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        return f'{lhs_code} {op}= {rhs_code};\n'

    def _print_Assign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, FunctionCall) and isinstance(rhs.class_type, InhomogeneousTupleType):
            self._temporary_args = [ObjectAddress(a) for a in lhs]
            code = self._print(rhs)
            self._temporary_args = []
            return code
        # Inhomogenous tuples are unravelled and therefore do not exist in the c printer
        if isinstance(rhs, (NumpyArray, PythonTuple)):
            return self.copy_NumpyArray_Data(lhs, rhs)
        if isinstance(rhs, (NumpySum, NumpyAmax, NumpyAmin)):
            return self._print(rhs)
        if isinstance(rhs, (NumpyFull)):
            return self.arrayFill(expr)
        lhs_code = self._print(lhs)
        if isinstance(rhs, (PythonList, PythonSet, PythonDict)):
            return self.init_stc_container(rhs, expr.lhs)
        rhs_code = self._print(rhs)
        return f'{lhs_code} = {rhs_code};\n'

    def _print_AliasAssign(self, expr):
        lhs_var = expr.lhs
        rhs_var = expr.rhs

        lhs_address = ObjectAddress(lhs_var)
        rhs_address = ObjectAddress(rhs_var)

        # The condition below handles the case of reassigning a pointer to an array view.
        if isinstance(lhs_var, Variable) and lhs_var.is_ndarray and not lhs_var.is_optional:
            lhs = self._print(lhs_var)

            if isinstance(rhs_var, Variable) and rhs_var.is_ndarray:
                lhs_ptr = self._print(lhs_address)
                rhs = self._print(rhs_address)
                rhs_type = self.get_c_type(rhs_var.class_type)
                slicing = ', '.join(['{c_ALL}']*lhs_var.rank)
                code = f'{lhs} = cspan_slice({rhs_type}, {rhs}, {slicing});\n'
                if lhs_var.order != rhs_var.order:
                    code += f'cspan_transpose({lhs_ptr});\n'
                return code
            else:
                rhs = self._print(rhs_var)
                return f'{lhs} = {rhs};\n'
        else:
            lhs = self._print(lhs_address)
            rhs = self._print(rhs_address)

            return f'{lhs} = {rhs};\n'

    def _print_For(self, expr):
        self.set_scope(expr.scope)

        iterable = expr.iterable
        indices = iterable.loop_counters

        if isinstance(iterable, (VariableIterator, DictItems, DictKeys)) and \
                isinstance(iterable.variable.class_type, (DictType, HomogeneousSetType, HomogeneousListType)):
            var = iterable.variable
            iterable_type = var.class_type
            counter = Variable(IteratorType(iterable_type), indices[0].name)
            c_type = self.get_c_type(iterable_type)
            iterable_code = self._print(var)
            for_code = f'c_foreach ({self._print(counter)}, {c_type}, {iterable_code})'
            tmp_ref = DottedVariable(VoidType(), 'ref', memory_handling='alias', lhs = counter)
            if isinstance(iterable, DictItems):
                assigns = [Assign(expr.target[0], DottedVariable(VoidType(), 'first', lhs = tmp_ref)),
                           Assign(expr.target[1], DottedVariable(VoidType(), 'second', lhs = tmp_ref))]
            elif isinstance(iterable, DictKeys):
                assigns = [Assign(expr.target[0], DottedVariable(VoidType(), 'first', lhs = tmp_ref))]
            else:
                assigns = [Assign(expr.target[0], tmp_ref)]
            additional_assign = CodeBlock(assigns)
        else:
            range_iterable = iterable.get_range()
            if indices:
                index = indices[0]
                if iterable.num_loop_counters_required and index.is_temp:
                    self.scope.insert_variable(index)
            else:
                index = expr.target[0]

            targets = iterable.get_assign_targets()
            additional_assign = CodeBlock([AliasAssign(i, t) if i.is_alias else Assign(i, t) \
                                    for i,t in zip(expr.target[-len(targets):], targets)])

            index_code = self._print(index)
            step = range_iterable.step
            start_code = self._print(range_iterable.start)
            stop_code  = self._print(range_iterable.stop )
            step_code  = self._print(range_iterable.step )

            # testing if the step is a value or an expression
            if is_literal_integer(step):
                op = '>' if int(step) < 0 else '<'
                stop_condition = f'{index_code} {op} {stop_code}'
            else:
                stop_condition = f'({step_code} > 0) ? ({index_code} < {stop_code}) : ({index_code} > {stop_code})'
            for_code = f'for ({index_code} = {start_code}; {stop_condition}; {index_code} += {step_code})\n'

        if self._additional_code:
            for_code = self._additional_code + for_code
            self._additional_code = ''

        body = self._print(additional_assign) + self._print(expr.body)

        self.exit_scope()
        return for_code + '{\n' + body + '}\n'

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs = expand_to_loops(expr,
                    self.scope.get_temporary_variable, self.scope,
                    language_has_vectors = False)
        else:
            body_exprs = expr.body
        body_stmts = []
        for b in body_exprs :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body_stmts.append(code)
        return ''.join(self._print(b) for b in body_stmts)

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return "M_E"

    def _print_Pi(self, expr):
        return 'M_PI'

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

    def _print_PythonReal(self, expr):
        return 'creal({})'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return 'cimag({})'.format(self._print(expr.internal_var))

    def _print_PythonConjugate(self, expr):
        return 'conj({})'.format(self._print(expr.internal_var))

    def _handle_is_operator(self, Op, expr):
        """
        Get the code to print an `is` or `is not` expression.

        Get the code to print an `is` or `is not` expression. These two operators
        function similarly so this helper function reduces code duplication.

        Parameters
        ----------
        Op : str
            The C operator representing "is" or "is not".

        expr : PyccelIs/PyccelIsNot
            The expression being printed.

        Returns
        -------
        str
            The code describing the expression.

        Raises
        ------
        PyccelError : Raised if the comparison is poorly defined.
        """

        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if Nil() in expr.args:
            lhs = ObjectAddress(expr.lhs) if isinstance(expr.lhs, Variable) else expr.lhs
            rhs = ObjectAddress(expr.rhs) if isinstance(expr.rhs, Variable) else expr.rhs

            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return '{} {} {}'.format(lhs, Op, rhs)

        if (a.dtype is PythonNativeBool() and b.dtype is PythonNativeBool()):
            return '{} {} {}'.format(lhs, Op, rhs)
        else:
            errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                          symbol=expr, severity='fatal')

    def _print_PyccelIsNot(self, expr):
        return self._handle_is_operator("!=", expr)

    def _print_PyccelIs(self, expr):
        return self._handle_is_operator("==", expr)

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond is not True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assign):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {\n" % self._print(c))
                elif i == len(expr.args) - 1 and c is True:
                    lines.append("else {\n")
                else:
                    lines.append("else if (%s) {\n" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}\n")
            return "".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Constant(self, expr):
        if expr == math_constants['inf']:
            self.add_import(c_imports['math'])
            return 'HUGE_VAL'
        elif expr == math_constants['nan']:
            self.add_import(c_imports['math'])
            return 'NAN'
        elif expr == math_constants['pi']:
            self.add_import(c_imports['math'])
            return 'M_PI'
        elif expr == math_constants['e']:
            self.add_import(c_imports['math'])
            return 'M_E'
        else:
            cast_func = DtypePrecisionToCastFunction[expr.dtype]
            return self._print(cast_func(expr.value))


    def _print_Variable(self, expr):
        if self.is_c_pointer(expr):
            return '(*{0})'.format(expr.name)
        else:
            return expr.name

    def _print_FunctionDefArgument(self, expr):
        return self._print(expr.name)

    def _print_FunctionCallArgument(self, expr):
        return self._print(expr.value)

    def _print_ObjectAddress(self, expr):
        obj_code = self._print(expr.obj)
        if isinstance(expr.obj, ObjectAddress) or not self.is_c_pointer(expr.obj):
            return f'&{obj_code}'
        else:
            if obj_code.startswith('(*') and obj_code.endswith(')'):
                return f'{obj_code[2:-1]}'
            else:
                return obj_code

    def _print_PointerCast(self, expr):
        declare_type = self.get_declare_type(expr.cast_type)
        if not self.is_c_pointer(expr.cast_type):
            declare_type += '*'
        obj = expr.obj
        if not isinstance(obj, ObjectAddress):
            obj = ObjectAddress(expr.obj)
        var_code = self._print(obj)
        return f'(*({declare_type})({var_code}))'

    def _print_Comment(self, expr):
        comments = self._print(expr.text)

        return '/*' + comments + '*/\n'

    def _print_Assert(self, expr):
        if isinstance(expr.test, LiteralTrue):
            if sys.version_info < (3, 9):
                return ''
            else:
                return '//' + ast.unparse(expr.python_ast) + '\n' #pylint: disable=no-member
        condition = self._print(expr.test)
        self.add_import(c_imports['assert'])
        return f"assert({condition});\n"

    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_CommentBlock(self, expr):
        txts = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+4):
            ln = 20
        top  = '/*' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '*/\n'
        ln = len(top)-4
        bottom = '/*' + '_'*ln + '*/\n'

        txts = ['/*' + t + ' '*(ln - len(t)) + '*/\n' for t in txts]

        body = ''.join(i for i in txts)

        return ''.join([top, body, bottom])

    def _print_EmptyNode(self, expr):
        return ''

    def _print_MaxLimit(self, expr):
        class_type = expr.class_type
        if isinstance(class_type.primitive_type, PrimitiveIntegerType):
            self.add_import(c_imports['stdint'])
            return f'INT{class_type.precision*8}_MAX'
        elif isinstance(class_type.primitive_type, PrimitiveFloatingPointType):
            self.add_import(c_imports['float'])
            numpy_type = numpy_precision_map[(class_type.primitive_type, class_type.precision)]
            if numpy_type is NumpyFloat32Type():
                return 'FLT_MAX'
            elif numpy_type is NumpyFloat64Type():
                return 'DBL_MAX'
            elif numpy_type is NumpyFloat128Type():
                return 'LDBL_MAX'
        raise errors.report(PYCCEL_INTERNAL_ERROR,
                symbol = expr, severity='fatal')

    def _print_MinLimit(self, expr):
        class_type = expr.class_type
        if isinstance(class_type.primitive_type, PrimitiveIntegerType):
            self.add_import(c_imports['stdint'])
            return f'INT{class_type.precision*8}_MIN'
        elif isinstance(class_type.primitive_type, PrimitiveFloatingPointType):
            self.add_import(c_imports['float'])
            numpy_type = numpy_precision_map[(class_type.primitive_type, class_type.precision)]
            if numpy_type is NumpyFloat32Type():
                return '-FLT_MAX'
            elif numpy_type is NumpyFloat64Type():
                return '-DBL_MAX'
            elif numpy_type is NumpyFloat128Type():
                return '-LDBL_MAX'
        raise errors.report(PYCCEL_INTERNAL_ERROR,
                symbol = expr, severity='fatal')

    #=================== OMP ==================

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            clauses = ' ' + expr.combined
        clauses += str(expr.txt)
        if expr.has_nowait:
            clauses = clauses + ' nowait'
        omp_expr = '#pragma omp {}{}\n'.format(expr.name, clauses)

        if expr.is_multiline:
            if expr.combined is None:
                omp_expr += '{\n'
            elif (expr.combined and "for" not in expr.combined):
                if ("masked taskloop" not in expr.combined) and ("distribute" not in expr.combined):
                    omp_expr += '{\n'

        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        return '}\n'
    #=====================================

    def _print_Program(self, expr):
        mod = expr.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
        self._current_module = mod
        self.set_scope(expr.scope)
        body  = self._print(expr.body)
        variables = self.scope.variables.values()
        decs = ''.join(self._print(Declare(v)) for v in variables)

        imports = [*expr.imports, *self._additional_imports.values()]
        imports = self.sort_imports(imports)
        imports = ''.join(self._print(i) for i in imports)

        self.exit_scope()
        self._current_module = None
        return ('{imports}'
                'int main()\n{{\n'
                '{decs}'
                '{body}'
                'return 0;\n'
                '}}').format(imports=imports,
                                    decs=decs,
                                    body=body)

    #================== CLASSES ==================

    def _print_CustomDataType(self, expr):
        return "struct " + expr.name

    def _print_Del(self, expr):
        return ''.join(self._print(var) for var in expr.variables)

    def _print_ClassDef(self, expr):
        methods = ''.join(self._print(method) for method in expr.methods)
        interfaces = ''.join(self._print(function) for interface in expr.interfaces for function in interface.functions)

        return methods + interfaces

    #================== List methods ==================
    def _print_ListAppend(self, expr):
        target = expr.list_obj
        class_type = target.class_type
        c_type = self.get_c_type(class_type)
        arg = self._print(expr.args[0])
        list_obj = self._print(ObjectAddress(expr.list_obj))
        return f'{c_type}_push({list_obj}, {arg});\n'

    def _print_ListPop(self, expr):
        class_type = expr.list_obj.class_type
        c_type = self.get_c_type(class_type)
        list_obj = self._print(ObjectAddress(expr.list_obj))
        if expr.index_element:
            self.add_import(Import('stc/vec', AsName(VariableTypeAnnotation(class_type), c_type)))
            if is_literal_integer(expr.index_element) and int(expr.index_element) < 0:
                idx_code = self._print(PyccelAdd(PythonLen(expr.list_obj), expr.index_element, simplify=True))
            else:
                idx_code = self._print(expr.index_element)
            return f'{c_type}_pull_elem({list_obj}, {idx_code})'
        else:
            return f'{c_type}_pull({list_obj})'

    def _print_ListInsert(self, expr):
        target = expr.list_obj
        class_type = target.class_type
        c_type = self.get_c_type(class_type)
        idx = self._print(expr.index)
        obj = self._print(expr.object)
        list_obj = self._print(ObjectAddress(expr.list_obj))
        return f'{c_type}_insert({list_obj}, {idx}, {obj});\n'

    def _print_ListClear(self, expr):
        target = expr.list_obj
        class_type = target.class_type
        c_type = self.get_c_type(class_type)
        list_obj = self._print(ObjectAddress(expr.list_obj))
        return f'{c_type}_clear({list_obj});\n'

    def _print_ListReverse(self, expr):
        class_type = expr.list_obj.class_type
        c_type = self.get_c_type(class_type)
        list_obj = self._print(ObjectAddress(expr.list_obj))
        self.add_import(Import('stc/algorithm' ,AsName(VariableTypeAnnotation(class_type), c_type)))
        return f'c_reverse({c_type}, {list_obj});\n'

    #================== Set methods ==================

    def _print_SetPop(self, expr):
        dtype = expr.set_variable.class_type
        var_type = self.get_c_type(dtype)
        self.add_import(Import('stc/hset', AsName(VariableTypeAnnotation(dtype), var_type)))
        set_var = self._print(ObjectAddress(expr.set_variable))
        # See pyccel/stdlib/STC_Extensions/Set_extensions.h for the definition
        return f'{var_type}_pop({set_var})'

    def _print_SetClear(self, expr):
        var_type = self.get_c_type(expr.set_variable.class_type)
        set_var = self._print(ObjectAddress(expr.set_variable))
        return f'{var_type}_clear({set_var});\n'

    def _print_SetAdd(self, expr):
        var_type = self.get_c_type(expr.set_variable.class_type)
        set_var = self._print(ObjectAddress(expr.set_variable))
        arg = self._print(expr.args[0])
        return f'{var_type}_push({set_var}, {arg});\n'

    def _print_SetCopy(self, expr):
        var_type = self.get_c_type(expr.set_variable.class_type)
        set_var = self._print(expr.set_variable)
        return f'{var_type}_clone({set_var})'

    def _print_SetUnion(self, expr):
        assign_base = expr.get_direct_user_nodes(lambda n: isinstance(n, Assign))
        if not assign_base:
            errors.report("The result of the union call must be saved into a variable",
                    severity='error', symbol=expr)
        class_type = expr.set_variable.class_type
        var_type = self.get_c_type(class_type)
        self.add_import(Import('stc/hset', AsName(VariableTypeAnnotation(class_type), var_type)))
        set_var = self._print(ObjectAddress(expr.set_variable))
        args = ', '.join([str(len(expr.args)), *(self._print(ObjectAddress(a)) for a in expr.args)])
        # See pyccel/stdlib/STC_Extensions/Set_extensions.h for the definition
        return f'{var_type}_union({set_var}, {args})'

    def _print_SetIntersectionUpdate(self, expr):
        class_type = expr.set_variable.class_type
        var_type = self.get_c_type(class_type)
        self.add_import(Import('stc/hset', AsName(VariableTypeAnnotation(class_type), var_type)))
        set_var = self._print(ObjectAddress(expr.set_variable))
        # See pyccel/stdlib/STC_Extensions/Set_extensions.h for the definition
        return ''.join(f'{var_type}_intersection_update({set_var}, {self._print(ObjectAddress(a))});\n' \
                for a in expr.args)

    def _print_SetDiscard(self, expr):
        var_type = self.get_c_type(expr.set_variable.class_type)
        set_var = self._print(ObjectAddress(expr.set_variable))
        arg_val = self._print(expr.args[0])
        return f'{var_type}_erase({set_var}, {arg_val});\n'

    def _print_SetIsDisjoint(self, expr):
        var_type = self.get_c_type(expr.set_variable.class_type)
        set_var = self._print(ObjectAddress(expr.set_variable))
        arg_val = self._print(ObjectAddress(expr.args[0]))
        # See pyccel/stdlib/STC_Extensions/Set_extensions.h for the definition
        return f'{var_type}_is_disjoint({set_var}, {arg_val});\n'

    def _print_PythonSet(self, expr):
        tmp_var = self.scope.get_temporary_variable(expr.class_type, shape = expr.shape,
                    memory_handling='heap')
        self._additional_code += self.init_stc_container(expr, tmp_var)
        return self._print(tmp_var)

    #================== Dict methods ==================

    def _print_DictClear(self, expr):
        c_type = self.get_c_type(expr.dict_obj.class_type)
        dict_var = self._print(ObjectAddress(expr.dict_obj))
        return f"{c_type}_clear({dict_var});\n"

    def _print_DictGet(self, expr):
        target = expr.dict_obj
        class_type = target.class_type
        c_type = self.get_c_type(class_type)

        dict_var = self._print(ObjectAddress(target))
        if isinstance(class_type.key_type, StringType):
            key = self._print(CStrStr(expr.key))
        else:
            key = self._print(expr.key)

        if expr.default_value is not None:
            default_val = self._print(expr.default_value)
            return f"({c_type}_contains({dict_var}, {key}) ? *{c_type}_at({dict_var}, {key}) : {default_val})"

        raise errors.report("Accessing dictionary using get without default is not supported yet", symbol=expr, severity="fatal")

    def _print_DictPop(self, expr):
        target = expr.dict_obj
        class_type = target.class_type
        c_type = self.get_c_type(class_type)

        dict_var = self._print(ObjectAddress(target))
        if isinstance(class_type.key_type, StringType):
            key = self._print(CStrStr(expr.key))
        else:
            key = self._print(expr.key)

        if expr.default_value is not None:
            default = self._print(expr.default_value)
            pop_expr = f"{c_type}_pop_with_default({dict_var}, {key}, {default})"
        else:
            pop_expr = f"{c_type}_pop({dict_var}, {key})"

        return pop_expr

    def _print_DictGetItem(self, expr):
        dict_obj = expr.dict_obj
        dict_obj_code = self._print(ObjectAddress(dict_obj))
        class_type = dict_obj.class_type
        container_type = self.get_c_type(class_type)
        if isinstance(class_type.key_type, StringType):
            key = self._print(CStrStr(expr.key))
        else:
            key = self._print(expr.key)
        assign = expr.get_user_nodes(Assign)
        if assign:
            assert len(assign) == 1
            assign_node = assign[0]
            lhs = assign_node.lhs
            if lhs == expr or lhs.is_user_of(expr):
                return f"(*{container_type}_at_mut({dict_obj_code}, {key}))"

        return f"(*{container_type}_at({dict_obj_code}, {key}))"

    #================== String methods ==================

    def _print_CStrStr(self, expr):
        arg = expr.args[0]
        code = self._print(ObjectAddress(arg))
        if code.startswith('&cstr_lit('):
            return code[10:-1]
        else:
            return f'cstr_str({code})'

    def _print_PythonStr(self, expr):
        arg = expr.args[0]
        arg_code = self._print(arg)
        if isinstance(arg.class_type, StringType):
            return f'cstr_clone({arg_code})'
        else:
            assert isinstance(arg.class_type, CharType) and getattr(arg, 'is_alias', True)
            return f'cstr_from({arg_code})'

    #=================== MACROS ==================

    def _print_MacroShape(self, expr):
        var = expr.argument
        if not isinstance(var, (Variable, IndexedElement)):
            raise TypeError('Expecting a variable, given {}'.format(type(var)))
        shape = var.shape

        if len(shape) == 1:
            shape = shape[0]


        elif not(expr.index is None):
            if expr.index < len(shape):
                shape = shape[expr.index]
            else:
                shape = '1'

        return self._print(shape)

    def _print_MacroCount(self, expr):

        var = expr.argument

        if var.rank == 0:
            return '1'
        else:
            return self._print(functools.reduce(
                lambda x,y: PyccelMul(x,y,simplify=True), var.shape))

    def _print_PrecomputedCode(self, expr):
        return expr.code

    def _print_AllDeclaration(self, expr):
        return ''

    def indent_code(self, code):
        """
        Add the necessary indentation to a string of code or a list of code lines.

        Add the necessary indentation to a string of code or a list of code lines.

        Parameters
        ----------
        code : str | iterable[str]
            The code which needs indenting.

        Returns
        -------
        str | list[str]
            The indented code. The type matches the type of the argument.
        """

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = " "*self._default_settings["tabwidth"]

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(line.endswith('{\n')) for line in code ]
        decrease = [ int(any(map(line.startswith, '}\n')))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            indent = tab*level
            pretty.append(f"{indent}{line}")
            level += increase[n]
        return pretty
