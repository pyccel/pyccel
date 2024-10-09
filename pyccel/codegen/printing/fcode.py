# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""


import functools
import string
import re
from collections import OrderedDict
from itertools import chain
from packaging.version import Version

import numpy as np

from pyccel.ast.basic import TypedAstNode

from pyccel.ast.bind_c import BindCPointer, BindCFunctionDef, BindCFunctionDefArgument, BindCModule, BindCClassDef

from pyccel.ast.builtins import PythonInt, PythonType, PythonPrint, PythonRange
from pyccel.ast.builtins import PythonTuple, DtypePrecisionToCastFunction
from pyccel.ast.builtins import PythonBool, PythonList, PythonSet

from pyccel.ast.core import FunctionDef, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core import SeparatorComment, Comment
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import FunctionCallArgument
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import Return, Module, For
from pyccel.ast.core import Import, CodeBlock, AsName, EmptyNode
from pyccel.ast.core import Assign, AliasAssign, Declare, Deallocate
from pyccel.ast.core import FunctionCall, PyccelFunctionDef

from pyccel.ast.datatypes import PrimitiveBooleanType, PrimitiveIntegerType, PrimitiveFloatingPointType, PrimitiveComplexType
from pyccel.ast.datatypes import SymbolicType, StringType, FixedSizeNumericType, HomogeneousContainerType
from pyccel.ast.datatypes import HomogeneousTupleType, HomogeneousListType, HomogeneousSetType, DictType
from pyccel.ast.datatypes import PythonNativeInt, PythonNativeBool
from pyccel.ast.datatypes import CustomDataType, InhomogeneousTupleType, TupleType
from pyccel.ast.datatypes import pyccel_type_to_original_type, PyccelType

from pyccel.ast.fortran_concepts import KindSpecification

from pyccel.ast.internals import Slice, PrecomputedCode, PyccelArrayShapeElement

from pyccel.ast.itertoolsext import Product

from pyccel.ast.literals  import LiteralInteger, LiteralFloat, Literal, LiteralEllipsis
from pyccel.ast.literals  import LiteralTrue, LiteralFalse, LiteralString
from pyccel.ast.literals  import Nil

from pyccel.ast.low_level_tools  import MacroDefinition, IteratorType, PairType

from pyccel.ast.mathext  import math_constants

from pyccel.ast.numpyext import NumpyEmpty, NumpyInt32
from pyccel.ast.numpyext import NumpyFloat, NumpyBool
from pyccel.ast.numpyext import NumpyReal, NumpyImag
from pyccel.ast.numpyext import NumpyRand, NumpyAbs
from pyccel.ast.numpyext import NumpyNewArray, NumpyArray
from pyccel.ast.numpyext import NumpyNonZero
from pyccel.ast.numpyext import NumpySign
from pyccel.ast.numpyext import NumpyIsFinite, NumpyIsNan

from pyccel.ast.numpytypes import NumpyNDArrayType

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelAnd
from pyccel.ast.operators import PyccelMod, PyccelNot, PyccelAssociativeParenthesis
from pyccel.ast.operators import PyccelUnarySub, PyccelLt, PyccelGt, IfTernaryOperator

from pyccel.ast.utilities import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities import expand_to_loops

from pyccel.ast.variable import Variable, IndexedElement, DottedName

from pyccel.parser.scope import Scope

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *
from pyccel.codegen.printing.codeprinter import CodePrinter

numpy_v1 = Version(np.__version__) < Version("2.0.0")

# TODO: add examples

__all__ = ["FCodePrinter", "fcode"]

numpy_ufunc_to_fortran = {
    'NumpyAbs'  : 'abs',
    'NumpyFabs'  : 'abs',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'Log',
    # 'NumpySqrt': 'Sqrt',  # sqrt is printed using _Print_NumpySqrt
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
    'NumpyIsFinite':'ieee_is_finite',
    'NumpyIsNan'  :'ieee_is_nan',
}

math_function_to_fortran = {
    'MathAcos'   : 'acos',
    'MathAcosh'  : 'acosh',
    'MathAsin'   : 'asin',
    'MathAsinh'  : 'asinh',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathAtanh'  : 'atanh',
    'MathCopysign': 'sign',
    'MathCos'    : 'cos',
    'MathCosh'   : 'cosh',
    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathExp'    : 'exp',
    # 'MathExpm1'  : '???', # TODO
    'MathFabs'   : 'abs',
    # 'MathFmod'   : '???',  # TODO
    # 'MathFsum'   : '???',  # TODO
    'MathGamma'  : 'gamma',
    'MathHypot'  : 'hypot',
    # 'MathLdexp'  : '???',  # TODO
    'MathLgamma' : 'log_gamma',
    'MathLog'    : 'log',
    'MathLog10'  : 'log10',
    # 'MathLog1p'  : '???', # TODO
    # 'MathLog2'   : '???', # TODO
    # 'MathPow'    : '???', # TODO
    'MathSin'    : 'sin',
    'MathSinh'   : 'sinh',
    'MathSqrt'   : 'sqrt',
    'MathTan'    : 'tan',
    'MathTanh'   : 'tanh',
    # ---
    'MathFloor'    : 'floor',
    # ---
    # 'MathIsclose' : '???', # TODO
    # 'MathIsfinite': '???', # TODO
    # 'MathIsinf'   : '???', # TODO
    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
    # --------------------------- cmath functions --------------------------
    'CmathAcos'  : 'acos',
    'CmathAcosh' : 'acosh',
    'CmathAsin'  : 'asin',
    'CmathAsinh' : 'asinh',
    'CmathAtan'  : 'atan',
    'CmathAtanh' : 'atanh',
    'CmathCos'   : 'cos',
    'CmathCosh'  : 'cosh',
    'CmathExp'   : 'exp',
    'CmathSin'   : 'sin',
    'CmathSinh'  : 'sinh',
    'CmathSqrt'  : 'sqrt',
    'CmathTan'   : 'tan',
    'CmathTanh'  : 'tanh',
}

INF = math_constants['inf']

_default_methods = {
    '__new__' : 'alloc',
    '__init__': 'create',
    '__del__' : 'free',
}

#==============================================================================
iso_c_binding = {
    PrimitiveIntegerType() : {
        1  : 'C_INT8_T',
        2  : 'C_INT16_T',
        4  : 'C_INT32_T',
        8  : 'C_INT64_T',
        16 : 'C_INT128_T'}, #not supported yet
    PrimitiveFloatingPointType() : {
        4  : 'C_FLOAT',
        8  : 'C_DOUBLE',
        16 : 'C_LONG_DOUBLE'}, #not supported yet
    PrimitiveComplexType() : {
        4  : 'C_FLOAT_COMPLEX',
        8  : 'C_DOUBLE_COMPLEX',
        16 : 'C_LONG_DOUBLE_COMPLEX'}, #not supported yet
    PrimitiveBooleanType() : {
        -1 : "C_BOOL"}
}

iso_c_binding_shortcut_mapping = {
    'C_INT8_T'              : 'i8',
    'C_INT16_T'             : 'i16',
    'C_INT32_T'             : 'i32',
    'C_INT64_T'             : 'i64',
    'C_INT128_T'            : 'i128',
    'C_FLOAT'               : 'f32',
    'C_DOUBLE'              : 'f64',
    'C_LONG_DOUBLE'         : 'f128',
    'C_FLOAT_COMPLEX'       : 'c32',
    'C_DOUBLE_COMPLEX'      : 'c64',
    'C_LONG_DOUBLE_COMPLEX' : 'c128',
    'C_BOOL'                : 'b1'
}

inc_keyword = (r'do\b', r'if\b',
               r'else\b', r'type\b\s*[^\(]',
               r'(elemental )?(pure )?(recursive )?((subroutine)|(function))\b',
               r'interface\b',r'module\b(?! *procedure)',r'program\b')
inc_regex = re.compile('|'.join('({})'.format(i) for i in inc_keyword))

end_keyword = ('do', 'if', 'type', 'function',
               'subroutine', 'interface','module','program')
end_regex_str = '(end ?({}))|(else)'.format('|'.join('({})'.format(k) for k in end_keyword))
dec_regex = re.compile(end_regex_str)

errors = Errors()

class FCodePrinter(CodePrinter):
    """
    A printer for printing code in Fortran.

    A printer to convert Pyccel's AST to strings of Fortran code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    printmethod = "_fcode"
    language = "Fortran"

    _default_settings = {
        'tabwidth': 2,
    }


    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename)

        super().__init__()
        self._constantImports = {}
        self._current_class    = None

        self._additional_code = None

        self.prefix_module = prefix_module

        self._generated_gFTL_extensions = {}

    def print_constant_imports(self):
        """
        Print the import of constant intrinsics.

        Print the import of constants such as `C_INT` from an intrinsic module (i.e. a
        module provided by Fortran) such as `iso_c_binding`.

        Returns
        -------
        str
            The code describing the import of the intrinsics.
        """
        macros = []
        for (name, imports) in self._constantImports.items():

            macro = f"use, intrinsic :: {name}, only : "
            rename = [c if isinstance(c, str) else c[0] + ' => ' + c[1] for c in imports]
            if len(rename) == 0:
                continue
            macro += " , ".join(rename)
            macro += "\n"
            macros.append(macro)
        return "".join(macros)

    def set_current_class(self, name):

        self._current_class = name

    def get_method(self, cls_name, method_name):
        container = self.scope
        while container:
            if cls_name in container.classes:
                cls = container.classes[cls_name]
                methods = cls.methods_as_dict
                if method_name in methods:
                    return methods[method_name]
                else:
                    interface_funcs = {f.name:f for i in cls.interfaces for f in i.functions}
                    if method_name in interface_funcs:
                        return interface_funcs[method_name]
                    errors.report(UNDEFINED_METHOD, symbol=method_name,
                        severity='fatal')
            container = container.parent_scope
        if isinstance(method_name, DottedName):
            return self.get_function(DottedName(method_name.name[1:]))
        errors.report(UNDEFINED_FUNCTION, symbol=method_name,
            severity='fatal')

    def get_function(self, name):
        container = self.scope
        while container:
            if name in container.functions:
                return container.functions[name]
            container = container.parent_scope
        if isinstance(name, DottedName):
            return self.get_function(name.name[-1])
        errors.report(UNDEFINED_FUNCTION, symbol=name,
            severity='fatal')

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def print_kind(self, expr):
        """
        Print the kind(precision) of a literal value or its shortcut if possible.

        Print the kind(precision) of a literal value or its shortcut if possible.

        Parameters
        ----------
        expr : TypedAstNode | PyccelType
            The object whose precision should be investigated.

        Returns
        -------
        str
            The code for the kind parameter.
        """
        dtype = expr if isinstance(expr, PyccelType) else expr.dtype

        constant_name = iso_c_binding[dtype.primitive_type][dtype.precision]

        constant_shortcut = iso_c_binding_shortcut_mapping[constant_name]
        if constant_shortcut not in self.scope.all_used_symbols and constant_name != constant_shortcut:
            self._constantImports.setdefault('ISO_C_Binding', set())\
                .add((constant_shortcut, constant_name))
            constant_name = constant_shortcut
        else:
            self._constantImports.setdefault('ISO_C_Binding', set())\
                .add(constant_name)
        return constant_name

    def _handle_inline_func_call(self, expr, assign_lhs = None):
        """
        Print a function call to an inline function.

        Use the arguments passed to an inline function to print
        its body with the passed arguments in place of the function
        arguments.

        Parameters
        ----------
        expr : FunctionCall
            The function call which should be printed inline.

        assign_lhs : List
            A list of lhs provided.

        Returns
        -------
        str
            The code for the inline function.
        """

        scope = self.scope
        func = expr.funcdef

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
        new_local_vars = [v.clone(self.scope.get_new_name(v.name)) \
                            for v in func.local_vars]
        for v in new_local_vars:
            self.scope.insert_variable(v)

        # Put functions into current scope
        for entry in ['variables', 'classes', 'functions']:
            self.scope.imports[entry].update(func.namespace_imports[entry])

        func.swap_in_args(args, new_local_vars)

        func.remove_presence_checks()

        body = func.body

        if len(func.results) == 0:
            # If there is no return then the code is already ok
            code = self._print(body)
        else:
            # Search for the return and replace it with an empty node
            result = body.get_attribute_nodes(Return)[0]
            empty_return = EmptyNode()
            body.substitute(result, empty_return, invalidate = False)

            # Everything before the return node needs handling before the line
            # which calls the inline function is executed
            code = self._print(body)
            if (not self._additional_code):
                self._additional_code = ''
            self._additional_code += code

            # Collect statements from results to return object
            if result.stmt:
                assigns = {i.lhs: i.rhs for i in result.stmt.body if isinstance(i, Assign)}
                self._additional_code += ''.join([self._print(i) for i in result.stmt.body if not isinstance(i, Assign)])
            else:
                assigns = {}

            # Put return statement back into function
            body.substitute(empty_return, result)

            if assign_lhs:
                assigns = [Assign(l, r) for l,r in zip(assign_lhs, assigns.values())]
                code = ''.join([self._print(a) for a in assigns])
            else:
                res_return_vars = [assigns.get(v,v) for v in result.expr]
                if len(res_return_vars) == 1:
                    return_val = res_return_vars[0]
                    parent_assign = return_val.get_direct_user_nodes(lambda x: isinstance(x, Assign))
                    if parent_assign:
                        return_val.remove_user_node(parent_assign[0], invalidate = False)
                        code = self._print(return_val)
                        return_val.set_current_user_node(parent_assign[0])
                    else:
                        code = self._print(return_val)
                else:
                    code = self._print(tuple(res_return_vars))

        # Put back original arguments
        func.reinstate_presence_checks()
        func.swap_out_args()

        for i in func.imports:
            self.add_import(i)
        if func.global_vars or func.global_funcs:
            mod = func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            current_mod = expr.get_user_nodes(Module, excluded_nodes=(FunctionCall,))[0]
            if current_mod is not mod:
                self.add_import(Import(mod.name, [AsName(v, v.name) \
                              for v in (*func.global_vars, *func.global_funcs)]))
                for v in (*func.global_vars, *func.global_funcs):
                    self.scope.insert_symbol(v.name)

        self.set_scope(scope)
        return code

    def _get_external_declarations(self, decs):
        """
        Find external functions and declare their result type.

        Look for any external functions in the local imports from
        the scope and use their definitions to create declarations
        from the results. These declarations are stored in the list
        passed as argument.

        Parameters
        ----------
        decs : list
            The list where the declarations necessary to use the external
            functions will be stored.
        """
        for key,f in self.scope.imports['functions'].items():
            if isinstance(f, FunctionDef) and f.is_external:
                v = f.results[0].var.clone(str(key))
                decs.append(Declare(v, external=True))

    def _calculate_class_names(self, expr):
        """
        Calculate the class names of the functions in a class.

        Calculate the names that will be referenced from the class
        for each function in a class. Also rename magic methods.

        Parameters
        ----------
        expr : ClassDef
            The class whose functions should be renamed.
        """
        scope = expr.scope
        name = expr.name.lower()
        for method in expr.methods:
            if not method.is_inline:
                m_name = method.name
                if m_name in _default_methods:
                    suggested_name = _default_methods[m_name]
                    scope.rename_function(method, suggested_name)
                method.cls_name = scope.get_new_name(f'{name}_{method.name}')
        for i in expr.interfaces:
            for f in i.functions:
                if not f.is_inline:
                    i_name = f.name
                    if i_name in _default_methods:
                        suggested_name = _default_methods[i_name]
                        scope.rename_function(f, suggested_name)
                    f.cls_name = scope.get_new_name(f'{name}_{f.name}')

    def _build_gFTL_module(self, expr_type):
        """
        Build the gFTL module to create container types.

        Create a module which will import the gFTL include files
        in order to create container types (e.g lists, sets, etc).
        The name of the module is derived from the name of the type.

        Parameters
        ----------
        expr_type : DataType
            The data type to be defined in a gFTL module.

        Returns
        -------
        Import
            The import which allows the new type to be accessed.
        """
        # Get the type used in the dict for compatible types (e.g. float vs float64)
        matching_expr_type = next((t for t in self._generated_gFTL_extensions if expr_type == t), None)
        if matching_expr_type:
            module = self._generated_gFTL_extensions[matching_expr_type]
            mod_name = module.name
        else:
            if isinstance(expr_type, HomogeneousListType):
                include = Import(LiteralString('vector/template.inc'), Module('_', (), ()))
                element_type = expr_type.element_type
                if isinstance(element_type, FixedSizeNumericType):
                    imports_and_macros = [MacroDefinition('T', element_type.primitive_type),
                              MacroDefinition('T_KINDLEN(context)', KindSpecification(element_type))]
                else:
                    imports_and_macros = [MacroDefinition('T', element_type)]
                if isinstance(element_type, (NumpyNDArrayType, HomogeneousTupleType)):
                    imports_and_macros.append(MacroDefinition('T_rank', element_type.rank))
                elif not isinstance(element_type, FixedSizeNumericType):
                    raise NotImplementedError("Support for lists of types defined in other modules is not yet implemented")
                imports_and_macros.append(MacroDefinition('Vector', expr_type))
                imports_and_macros.append(MacroDefinition('VectorIterator', IteratorType(expr_type)))
            elif isinstance(expr_type, HomogeneousSetType):
                include = Import(LiteralString('set/template.inc'), Module('_', (), ()))
                element_type = expr_type.element_type
                imports_and_macros = []
                if isinstance(element_type, FixedSizeNumericType):
                    tmpVar_x = Variable(element_type, 'x')
                    tmpVar_y = Variable(element_type, 'y')
                    if isinstance(element_type.primitive_type, PrimitiveComplexType):
                        complex_tool_import = Import('pyc_tools_f90', Module('pyc_tools_f90',(),()))
                        self.add_import(complex_tool_import)
                        imports_and_macros.append(complex_tool_import)
                        compare_func = FunctionDef('complex_comparison',
                                                   [FunctionDefArgument(tmpVar_x), FunctionDefArgument(tmpVar_y)],
                                                   [FunctionDefResult(Variable(PythonNativeBool(), 'c'))], [])
                        lt_def = FunctionCall(compare_func, [tmpVar_x, tmpVar_y])
                    else:
                        lt_def = PyccelAssociativeParenthesis(PyccelLt(tmpVar_x, tmpVar_y))
                    imports_and_macros.extend([MacroDefinition('T', element_type.primitive_type),
                              MacroDefinition('T_KINDLEN(context)', KindSpecification(element_type)),
                              MacroDefinition('T_LT(x,y)', lt_def)])
                else:
                    raise NotImplementedError("Support for sets of types which define their own < operator is not yet implemented")
                imports_and_macros.append(MacroDefinition('Set', expr_type))
                imports_and_macros.append(MacroDefinition('SetIterator', IteratorType(expr_type)))
            elif isinstance(expr_type, DictType):
                include = Import(LiteralString('map/template.inc'), Module('_', (), ()))
                key_type = expr_type.key_type
                value_type = expr_type.value_type
                imports_and_macros = []
                if isinstance(key_type, FixedSizeNumericType):
                    tmpVar_x = Variable(key_type, 'x')
                    tmpVar_y = Variable(key_type, 'y')
                    if isinstance(key_type.primitive_type, PrimitiveComplexType):
                        complex_tool_import = Import('pyc_tools_f90', Module('pyc_tools_f90',(),()))
                        self.add_import(complex_tool_import)
                        imports_and_macros.append(complex_tool_import)
                        compare_func = FunctionDef('complex_comparison',
                                                   [FunctionDefArgument(tmpVar_x), FunctionDefArgument(tmpVar_y)],
                                                   [FunctionDefResult(Variable(PythonNativeBool(), 'c'))], [])
                        lt_def = FunctionCall(compare_func, [tmpVar_x, tmpVar_y])
                    else:
                        lt_def = PyccelAssociativeParenthesis(PyccelLt(tmpVar_x, tmpVar_y))
                    imports_and_macros.extend([MacroDefinition('Key', key_type.primitive_type),
                                   MacroDefinition('Key_KINDLEN(context)', KindSpecification(key_type)),
                                   MacroDefinition('Key_LT(x,y)', lt_def)])
                else:
                    raise NotImplementedError("Support for dicts whose keys define their own < operator is not yet implemented")
                if isinstance(value_type, FixedSizeNumericType):
                    imports_and_macros.extend([MacroDefinition('T', value_type.primitive_type),
                                   MacroDefinition('T_KINDLEN(context)', KindSpecification(value_type))])
                else:
                    raise NotImplementedError(f"Support for dictionary values of type {value_type} not yet implemented")
                imports_and_macros.append(MacroDefinition('Pair', PairType(key_type, value_type)))
                imports_and_macros.append(MacroDefinition('Map', expr_type))
                imports_and_macros.append(MacroDefinition('MapIterator', IteratorType(expr_type)))
            else:
                raise NotImplementedError(f"Unkown gFTL import for type {expr_type}")

            typename = self._print(expr_type)
            mod_name = f'{typename}_mod'
            module = Module(mod_name, (), (), scope = Scope(), imports = [*imports_and_macros, include],
                                       is_external = True)

            self._generated_gFTL_extensions[expr_type] = module

        return Import(f'gFTL_extensions/{mod_name}', module)

    def _get_node_without_gFTL(self, expr):
        """
        Get the code to print an AST node without using gFTL.

        This function ensures that lists are printed as basic Fortran lists instead of using a
        gFTL Vector. This is useful when lists are used as arguments to intrinsic functions.

        Parameters
        ----------
        expr : PyccelAstType
            The element of the array.

        Returns
        -------
        str
            The code for the element.
        """
        if isinstance(expr, (PythonList, PythonTuple)):
            shape = tuple(reversed(expr.shape))
            if len(shape)>1:
                elements = ', '.join(self._get_node_without_gFTL(i) for i in expr)
                shape    = ', '.join(self._print(i) for i in shape)
                return 'reshape(['+ elements + '], [' + shape + '])'
            args = ', '.join(self._print(f) for f in expr)
            return f'[{args}]'
        else:
            return self._print(expr)

    def _apply_cast(self, target_type, *args):
        """
        Cast the arguments to the specified target type.

        Cast the arguments to the specified target type. For literal containers this
        function applies the cast to the elements.

        Parameters
        ----------
        target_type : PyccelType
            The type which we should cast to.
        *args : TypedAstNode
            A node that should be cast to the target type.

        Returns
        -------
        TypedAstNode | iterable[TypedAstNode]
            A TypedAstNode for each argument. The new nodes will have the target type.
        """
        try :
            cast_func = DtypePrecisionToCastFunction[target_type]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        new_args = []
        for a in args:
            if target_type != a.class_type:
                if isinstance(a, (PythonList, PythonSet, PythonTuple)):
                    container = type(a)
                    a = container(*[self._apply_cast(target_type, ai) for ai in a])
                else:
                    a = cast_func(a)
            new_args.append(a)

        if len(args) == 1:
            return new_args[0]
        else:
            return new_args

    # ============ Elements ============ #
    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        name = self._print(expr.name)
        name = name.replace('.', '_')
        if not name.startswith('mod_') and self.prefix_module:
            name = f'{self.prefix_module}_{name}'

        imports = ''.join(self._print(i) for i in expr.imports)

        # Define declarations
        decs = ''
        # ...
        for c in expr.classes:
            if not isinstance(c, BindCClassDef):
                self._calculate_class_names(c)

        class_decs_and_methods = [self._print(i) for i in expr.classes]
        decs += '\n'.join(c[0] for c in class_decs_and_methods)
        # ...

        declarations = list(expr.declarations)
        # look for external functions and declare their result type
        self._get_external_declarations(declarations)
        decs += ''.join(self._print(d) for d in declarations)

        # ... TODO add other elements
        private_funcs = [f.name for f in expr.funcs if f.is_private]
        private = private_funcs
        if private:
            private = ','.join(self._print(i) for i in private)
            private = 'private :: {}\n'.format(private)
        else:
            private = ''
        # ...

        # ...
        sep = self._print(SeparatorComment(40))
        interfaces = ''
        if expr.interfaces and not isinstance(expr, BindCModule):
            interfaces = '\n'.join(self._print(i) for i in expr.interfaces)

        func_strings = []
        # Get class functions
        func_strings += [c[1] for c in class_decs_and_methods]
        if expr.funcs:
            func_strings += [''.join([sep, self._print(i), sep]) for i in expr.funcs]
        if isinstance(expr, BindCModule):
            func_strings += [''.join([sep, self._print(i), sep]) for i in expr.variable_wrappers]
        body = '\n'.join(func_strings)
        # ...

        contains = 'contains\n' if (expr.funcs or expr.classes or expr.interfaces) else ''
        imports += ''.join(self._print(i) for i in self._additional_imports.values())
        imports = self.print_constant_imports() + imports
        implicit_none = '' if expr.is_external else 'implicit none\n'

        parts = ['module {}\n'.format(name),
                 imports,
                 implicit_none,
                 private,
                 decs,
                 interfaces,
                 contains,
                 body,
                 'end module {}\n'.format(name)]

        self.exit_scope()

        return '\n'.join([a for a in parts if a])

    def _print_Program(self, expr):
        self.set_scope(expr.scope)

        name    = 'prog_{0}'.format(self._print(expr.name)).replace('.', '_')
        imports = ''.join(self._print(i) for i in expr.imports)
        body    = self._print(expr.body)

        # Print the declarations of all variables in the scope, which include:
        #  - user-defined variables (available in Program.variables)
        #  - pyccel-generated variables added to Scope when printing 'expr.body'
        variables = self.scope.variables.values()
        decs = ''.join(self._print(Declare(v)) for v in variables)

        # Detect if we are using mpi4py
        # TODO should we find a better way to do this?
        mpi = any('mpi4py' == str(getattr(i.source, 'name', i.source)) for i in expr.imports)

        # Additional code and variable declarations for MPI usage
        # TODO: check if we should really add them like this
        if mpi:
            body = 'call mpi_init(ierr)\n'+\
                   '\nallocate(status(0:-1 + mpi_status_size)) '+\
                   '\nstatus = 0\n'+\
                   body +\
                   '\ncall mpi_finalize(ierr)'

            decs += '\ninteger :: ierr = -1' +\
                    '\ninteger, allocatable :: status (:)'
        imports += ''.join(self._print(i) for i in self._additional_imports.values())
        imports += "\n" + self.print_constant_imports()
        parts = ['program {}\n'.format(name),
                 imports,
                'implicit none\n',
                 decs,
                 body,
                'end program {}\n'.format(name)]

        self.exit_scope()

        return '\n'.join(a for a in parts if a)

    def _print_Import(self, expr):

        source = ''
        if expr.ignore:
            return ''

        source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1]
        elif isinstance(source, LiteralString):
            source = source.python_value
        else:
            source = self._print(source)

        # importing of pyccel extensions is not printed
        if source in pyccel_builtin_import_registry:
            return ''

        if source.endswith('.inc'):
            return f'#include <{source}>\n'

        if expr.source_module:
            source = expr.source_module.name

        if 'mpi4py' == str(getattr(expr.source,'name',expr.source)):
            return 'use mpi\n' + 'use mpiext\n'

        targets = [t for t in expr.target if not isinstance(t.object, Module)]

        if len(targets) == 0:
            return f'use {source}\n'

        targets = [t for t in targets if not getattr(t.object, 'is_inline', False)]
        if len(targets) == 0:
            return ''

        prefix = f'use {source}, only:'

        code = ''
        for i in targets:
            old_name = i.name
            new_name = i.local_alias
            if old_name != new_name:
                target = '{target} => {name}'.format(target=new_name,
                                                     name=old_name)
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(new_name, DottedName):
                target = '_'.join(self._print(j) for j in new_name.name)
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=target)

            elif isinstance(new_name, str):
                line = '{prefix} {target}'.format(prefix=prefix,
                                                  target=new_name)

            else:
                raise TypeError('Expecting str, PyccelSymbol, DottedName or AsName, '
                                'given {}'.format(type(i)))

            code = (code + '\n' + line) if code else line

        # in some cases, the source is given as a string (when using metavar)
        code = code.replace("'", '')
        return code + '\n'

    def _print_PythonPrint(self, expr):
        end = LiteralString('\n')
        sep = LiteralString(' ')
        code = ''
        empty_end = FunctionCallArgument(LiteralString(''), 'end')
        space_end = FunctionCallArgument(LiteralString(' '), 'end')
        empty_sep = FunctionCallArgument(LiteralString(''), 'sep')
        for f in expr.expr:
            if f.has_keyword:
                if f.keyword == 'sep':
                    sep = f.value
                elif f.keyword == 'end':
                    end = f.value
                else:
                    errors.report("{} not implemented as a keyworded argument".format(f.keyword), severity='fatal')
        args_format = []
        args = []
        orig_args = [f for f in expr.expr if not f.has_keyword]
        separator = self._print(sep)


        tuple_start = FunctionCallArgument(LiteralString('('))
        tuple_sep   = LiteralString(', ')
        tuple_end   = FunctionCallArgument(LiteralString(')'))

        for i, f in enumerate(orig_args):
            if f.keyword:
                continue
            else:
                f = f.value
            if isinstance(f, (PythonTuple, PythonList, str)):
                if args_format:
                    code += self._formatted_args_to_print(args_format, args, sep, separator, expr)
                    args_format = []
                    args = []
                if i + 1 == len(orig_args):
                    end_of_tuple = empty_end
                else:
                    end_of_tuple = FunctionCallArgument(sep, 'end')
                args = [FunctionCallArgument(print_arg) for tuple_elem in f for print_arg in (tuple_elem, tuple_sep)][:-1]
                if len(f) == 1:
                    args.append(FunctionCallArgument(LiteralString(',')))
                code += self._print(PythonPrint([tuple_start, *args, tuple_end, empty_sep, end_of_tuple], file=expr.file))
                args = []
            elif isinstance(f, PythonType):
                args_format.append('A')
                args.append(self._print(f.print_string))
            elif isinstance(f.class_type, (TupleType, HomogeneousContainerType)) and not isinstance(f.class_type, StringType) \
                    and not isinstance(f, FunctionCall):
                if args_format:
                    code += self._formatted_args_to_print(args_format, args, sep, separator, expr)
                    args_format = []
                    args = []
                loop_scope = self.scope.create_new_loop_scope()
                for_index = self.scope.get_temporary_variable(PythonNativeInt(), name='i')
                max_index = PyccelMinus(f.shape[0], LiteralInteger(1), simplify=True)
                for_range = PythonRange(max_index)
                print_body = [FunctionCallArgument(f[for_index])]
                if f.rank == 1:
                    print_body.append(space_end)

                for_body = [PythonPrint(print_body, file=expr.file)]
                for_loop = For(for_index, for_range, for_body, scope=loop_scope)
                for_end_char = LiteralString(']')
                for_end = FunctionCallArgument(for_end_char,
                                               keyword='end')

                body = CodeBlock([PythonPrint([FunctionCallArgument(LiteralString('[')), empty_end],
                                                file=expr.file),
                                  for_loop,
                                  PythonPrint([FunctionCallArgument(f[max_index]), for_end],
                                                file=expr.file)],
                                 unravelled=True)
                code += self._print(body)
            else:
                arg_format, arg = self._get_print_format_and_arg(f)
                args_format.append(arg_format)
                args.append(arg)
        code += self._formatted_args_to_print(args_format, args, end, separator, expr)
        return code

    def _formatted_args_to_print(self, fargs_format, fargs, fend, fsep, expr):
        """
        Produce a write statement from all necessary information.

        Produce a write statement from a list of formats, arguments, an end
        statement, and a separator.

        Parameters
        ----------
        fargs_format : iterable
            The format strings for the objects described by fargs.
        fargs : iterable
            The arguments to be printed.
        fend : TypedAstNode
            The character describing the end of the line.
        fsep : TypedAstNode
            The character describing the separator between elements.
        expr : TypedAstNode
            The PythonPrint currently printed.

        Returns
        -------
        str
            The Fortran code describing the write statement.
        """
        if fargs_format == ['*']:
            # To print the result of a FunctionCall
            return ', '.join(['print *', *fargs]) + '\n'


        args_list = [a_c if a_c != '' else "''" for a_c in fargs]
        fend_code = self._print(fend)
        advance = "yes" if fend_code == 'ACHAR(10)' else "no"

        if fsep != '':
            fargs_format = [af for a in fargs_format for af in (a, 'A')][:-1]
            args_list    = [af for a in args_list for af in (a, fsep)][:-1]

        if fend_code not in ('ACHAR(10)', ''):
            fargs_format.append('A')
            args_list.append(fend_code)

        args_code       = ' , '.join(args_list)
        args_formatting = ', '.join(fargs_format)
        if expr.file == "stderr":
            self._constantImports.setdefault('ISO_FORTRAN_ENV', set())\
                .add(("stderr", "error_unit"))
            return f"write(stderr, '({args_formatting})', advance=\"{advance}\") {args_code}\n"
        self._constantImports.setdefault('ISO_FORTRAN_ENV', set())\
                .add(("stdout", "output_unit"))
        return f"write(stdout, '({args_formatting})', advance=\"{advance}\") {args_code}\n"

    def _get_print_format_and_arg(self, var, var_code = None):
        """
        Get the format string and the printable argument for an object.

        Get the format string and the printable argument for an object.
        In other words get arg_format and arg such that var can be printed
        by doing:

        > write(*, arg_format) arg

        Parameters
        ----------
        var : TypedAstNode
            The object to be printed.
        var_code : str, optional
            The code which will print the variable (this is mostly useful when calling
            this function recursively, e.g. to print an inhomogenoeus tuple of function
            call results).

        Returns
        -------
        arg_format : str
            The format string.
        arg : str
            The Fortran code which represents var.
        """
        if var_code is None:
            var_code = self._print(var)
        arg = var_code

        var_type = var.dtype
        if isinstance(var.class_type, StringType):
            arg_format = 'A'
        elif isinstance(var, FunctionCall) and len(var.funcdef.results)>1 or \
                isinstance(var.class_type, InhomogeneousTupleType):
            var_elem_code = var_code[1:-1].split(', ')
            args_and_formats = [self._get_print_format_and_arg(v.var, c) for v,c in zip(var.funcdef.results, var_elem_code)]
            formats = ',", ",'.join(af[0] for af in args_and_formats)
            arg_format = f'"(",{formats},")"'
            arg = ', '.join(af[1] for af in args_and_formats)
        elif isinstance(var_type, FixedSizeNumericType):
            if isinstance(var_type.primitive_type, PrimitiveComplexType):
                float_format, real_arg = self._get_print_format_and_arg(NumpyReal(var))
                imag_arg = self._print(NumpyImag(var))
                arg_format = f'"(",{float_format}," + ",{float_format},"j)"'
                arg = f'{real_arg}, {imag_arg}'
            elif isinstance(var_type.primitive_type, PrimitiveFloatingPointType):
                dps = np.finfo(pyccel_type_to_original_type[var_type]).precision
                arg_format = f'F0.{dps}'
            elif isinstance(var_type.primitive_type, PrimitiveIntegerType):
                arg_format = 'I0'
            elif isinstance(var_type.primitive_type, PrimitiveBooleanType):
                arg_format = 'A'
                if isinstance(var, LiteralTrue):
                    arg = "'True'"
                elif isinstance(var, LiteralFalse):
                    arg = "'False'"
                else:
                    arg = f'merge("True ", "False", {var_code})'
            else:
                errors.report(f"Printing {var_type} type is not supported currently", severity='fatal')
        else:
            errors.report(f"Printing {var_type} type is not supported currently", severity='fatal')

        return arg_format, arg

    def _print_SymbolicPrint(self, expr):
        # for every expression we will generate a print
        code = '\n'.join(f"print *, 'sympy> {a}'" for a in expr.expr)
        return code + '\n'

    def _print_Comment(self, expr):
        comments = self._print(expr.text)
        return '!' + comments + '\n'

    def _print_CommentBlock(self, expr):
        txts   = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+2):
            ln = 20
        top  = '!' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '!'
        ln = len(top) - 2
        bottom = '!' + '_'*ln + '!'

        txts = ['!' + txt + ' '*(ln - len(txt)) + '!' for txt in txts]

        body = '\n'.join(i for i in txts)

        return ('{0}\n'
                '{1}\n'
                '{2}\n').format(top, body, bottom)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_AnnotatedComment(self, expr):
        accel = self._print(expr.accel)
        txt   = str(expr.txt)
        return '!${0} {1}\n'.format(accel, txt)

    def _print_tuple(self, expr):
        if expr[0].rank>0:
            raise NotImplementedError(' tuple with elements of rank > 0 is not implemented')
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_PythonAbs(self, expr):
        arg_code = self._get_node_without_gFTL(expr.arg)
        return f"abs({arg_code})"

    def _print_PythonTuple(self, expr):
        shape = tuple(reversed(expr.shape))
        if len(shape)>1:
            elements = ', '.join(self._print(i) for i in expr)
            shape    = ', '.join(self._print(i) for i in shape)
            return 'reshape(['+ elements + '], [' + shape + '])'
        args = ', '.join(self._print(f) for f in expr)
        return f'[{args}]'

    def _print_PythonList(self, expr):
        if len(expr.args) == 0:
            list_arg = ''
            assign = expr.get_direct_user_nodes(lambda a : isinstance(a, Assign))
            if assign:
                vec_type = self._print(assign[0].lhs.class_type)
            else:
                raise errors.report("Can't use an empty list without assigning it to a variable as the type cannot be deduced",
                        severity='fatal', symbol=expr)

        else:
            list_arg = self._print_PythonTuple(expr)
            vec_type = self._print(expr.class_type)
        return f'{vec_type}({list_arg})'

    def _print_PythonSet(self, expr):
        if len(expr.args) == 0:
            list_arg = ''
            assign = expr.get_direct_user_nodes(lambda a : isinstance(a, Assign))
            if assign:
                set_type = self._print(assign[0].lhs.class_type)
            else:
                raise errors.report("Can't use an empty set without assigning it to a variable as the type cannot be deduced",
                        severity='fatal', symbol=expr)

        else:
            list_arg = self._print_PythonTuple(expr)
            set_type = self._print(expr.class_type)
        return f'{set_type}({list_arg})'

    def _print_PythonDict(self, expr):
        if len(expr) == 0:
            list_arg = ''
            assign = expr.get_direct_user_nodes(lambda a : isinstance(a, Assign))
            if assign:
                dict_type = self._print(assign[0].lhs.class_type)
            else:
                raise errors.report("Can't use an empty dict without assigning it to a variable as the type cannot be deduced",
                        severity='fatal', symbol=expr)

        else:
            class_type = expr.class_type
            pair_type = self._print(PairType(class_type.key_type, class_type.value_type))
            args = ', '.join(f'{pair_type}({self._print(k)}, {self._print(v)})' for k,v in expr)
            list_arg = f'[{args}]'
            dict_type = self._print(class_type)
        return f'{dict_type}({list_arg})'

    def _print_InhomogeneousTupleVariable(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_FunctionDefArgument(self, expr):
        return self._print(expr.name)

    def _print_FunctionCallArgument(self, expr):
        if expr.keyword:
            return '{} = {}'.format(expr.keyword, self._print(expr.value))
        else:
            return '{}'.format(self._print(expr.value))

    def _print_Constant(self, expr):
        if expr == math_constants['nan']:
            errors.report("Can't print nan in Fortran",
                    severity='error', symbol=expr)
        val = LiteralFloat(expr.value)
        return self._print(val)

    def _print_DottedVariable(self, expr):
        if isinstance(expr.lhs, FunctionCall):
            base = expr.lhs.funcdef.results[0].var
            if (not self._additional_code):
                self._additional_code = ''
            var_name = self.scope.get_new_name()
            var = base.clone(var_name)

            self.scope.insert_variable(var)

            self._additional_code = self._additional_code + self._print(Assign(var,expr.lhs)) + '\n'
            return self._print(var) + '%' +self._print(expr.name)
        else:
            return self._print(expr.lhs) + '%' +self._print(expr.name)

    def _print_DottedName(self, expr):
        return ' % '.join(self._print(n) for n in expr.name)

    def _print_Lambda(self, expr):
        return '"{args} -> {expr}"'.format(args=expr.variables, expr=expr.expr)

    def _print_PythonSum(self, expr):
        args = ", ".join(self._get_node_without_gFTL(arg) for arg in expr.args)
        return f"sum({args})"

    def _print_PythonReal(self, expr):
        value = self._print(expr.internal_var)
        return f'real({value})'

    def _print_PythonImag(self, expr):
        value = self._print(expr.internal_var)
        return f'aimag({value})'

    #========================== List Methods ===============================#

    def _print_ListAppend(self, expr):
        target = expr.list_obj
        arg = self._print(expr.args[0])
        return f'call {target} % push_back({arg})\n'

    #========================== Set Methods ================================#

    def _print_SetAdd(self, expr):
        var = self._print(expr.set_variable)
        insert_obj = self._print(expr.args[0])
        return f'call {var} % insert( {insert_obj} )\n'

    def _print_SetClear(self, expr):
        var = self._print(expr.set_variable)
        return f'call {var} % clear()\n'

    #========================== Numpy Elements ===============================#

    def _print_NumpySum(self, expr):
        arg_code = self._get_node_without_gFTL(expr.arg)
        dtype = expr.arg.dtype.primitive_type
        if isinstance(dtype, PrimitiveBooleanType):
            return f'count({arg_code})'
        return f'sum({arg_code})'

    def _print_NumpyProduct(self, expr):
        arg_code = self._get_node_without_gFTL(expr.arg)
        return f'product({arg_code})'

    def _print_NumpyMatmul(self, expr):
        """Fortran print."""
        a_code = self._print(expr.a)
        b_code = self._print(expr.b)

        if expr.rank == 0:
            if isinstance(expr.a.dtype.primitive_type, PrimitiveBooleanType):
                a_code = self._print(PythonInt(expr.a))
            if isinstance(expr.b.dtype.primitive_type, PrimitiveBooleanType):
                b_code = self._print(PythonInt(expr.b))
            return 'sum({}*{})'.format(a_code, b_code)
        if expr.a.order and expr.b.order:
            if expr.a.order != expr.b.order:
                raise NotImplementedError("Mixed order matmul not supported.")

        # Fortran ordering
        if expr.a.order == 'F':
            return 'matmul({0},{1})'.format(a_code, b_code)

        # C ordering
        return 'matmul({1},{0})'.format(a_code, b_code)

    def _print_NumpyEmpty(self, expr):
        errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION, symbol=expr, severity='fatal')

    def _print_NumpyNorm(self, expr):
        arg = NumpyAbs(expr.arg) if isinstance(expr.arg.dtype.primitive_type, PrimitiveComplexType) else expr.arg
        arg_code = self._get_node_without_gFTL(arg)
        if expr.axis:
            axis = expr.axis
            if arg.order != 'F':
                axis = PyccelMinus(LiteralInteger(arg.rank), expr.axis, simplify=True)
            else:
                axis = LiteralInteger(expr.axis.python_value + 1)
            code = f'Norm2({arg_code},{self._print(axis)})'
        else:
            code = f'Norm2({arg_code})'

        return code

    def _print_NumpyLinspace(self, expr):

        if expr.stop.dtype != expr.dtype:
            st = self._apply_cast(expr.dtype, expr.stop)
            v = self._print(st)
        else:
            v = self._print(expr.stop)

        if not isinstance(expr.endpoint, LiteralFalse):
            lhs = expr.get_user_nodes(Assign)[0].lhs


            if expr.rank > 1:
                #expr.rank > 1, we need to replace the last index of the loop with the last index of the array.
                lhs_source = expr.get_user_nodes(Assign)[0].lhs
                lhs_source.substitute(expr.ind, PyccelMinus(expr.num, LiteralInteger(1), simplify = True))
                lhs = self._print(lhs_source)
            else:
                #Since the expr.rank == 1, we modify the last element in the array.
                lhs = self._print(IndexedElement(lhs,
                                                 PyccelMinus(expr.num, LiteralInteger(1),
                                                 simplify = True)))

            if isinstance(expr.endpoint, LiteralTrue):
                cond_template = lhs + ' = {stop}'
            else:
                cond_template = lhs + ' = merge({stop}, {lhs}, ({cond}))'
        if expr.rank > 1:
            template = '({start} + {index}*{step})'
            var = expr.ind
        else:
            template = '[(({start} + {index}*{step}), {index} = {zero},{end})]'
            var = self.scope.get_temporary_variable(PythonNativeInt(), 'linspace_index')

        init_value = template.format(
            start = self._print(expr.start),
            step  = self._print(expr.step),
            index = self._print(var),
            zero  = self._print(LiteralInteger(0)),
            end   = self._print(PyccelMinus(expr.num, LiteralInteger(1), simplify = True)),
        )

        if isinstance(expr.endpoint, LiteralFalse):
            code = init_value
        elif isinstance(expr.endpoint, LiteralTrue):
            code = init_value + '\n' + cond_template.format(stop=v)
        else:
            code = init_value + '\n' + cond_template.format(stop=v, lhs=lhs, cond=self._print(expr.endpoint))

        return code

    def _print_NumpyNonZeroElement(self, expr):

        ind   = self._print(self.scope.get_temporary_variable(PythonNativeInt()))
        array = expr.array

        if isinstance(array.dtype.primitive_type, PrimitiveBooleanType):
            mask  = self._print(array)
        else:
            mask  = self._print(NumpyBool(array))

        my_range = self._print(PythonRange(array.shape[expr.dim]))

        stmt  = 'pack([({ind}, {ind}={my_range})], {mask})'.format(
                ind = ind, mask = mask, my_range = my_range)

        return stmt

    def _print_NumpyCountNonZero(self, expr):

        axis  = expr.axis
        array = expr.array

        if isinstance(array.dtype.primitive_type, PrimitiveBooleanType):
            mask  = self._print(array)
        else:
            mask  = self._print(NumpyBool(array))

        kind  = self.print_kind(expr)

        if axis is None:
            stmt = 'count({}, kind = {})'.format(mask, kind)

            if expr.keep_dims.python_value:
                if expr.order == 'C':
                    shape    = ', '.join(self._print(i) for i in reversed(expr.shape))
                else:
                    shape    = ', '.join(self._print(i) for i in expr.shape)
                stmt = 'reshape([{}], [{}])'.format(stmt, shape)
        else:
            if array.order == 'C':
                f_dim = PyccelMinus(LiteralInteger(array.rank), expr.axis, simplify=True)
            else:
                f_dim = PyccelAdd(expr.axis, LiteralInteger(1), simplify=True)

            dim   = self._print(f_dim)
            stmt = 'count({}, dim = {}, kind = {})'.format(mask, dim, kind)

            if expr.keep_dims.python_value:

                if expr.order == 'C':
                    shape    = ', '.join(self._print(i) for i in reversed(expr.shape))
                else:
                    shape    = ', '.join(self._print(i) for i in expr.shape)
                stmt = 'reshape([{}], [{}])'.format(stmt, shape)

        return stmt

    def _print_NumpyWhere(self, expr):
        value_true, value_false = self._apply_cast(expr.dtype, expr.value_true, expr.value_false)

        condition   = self._print(expr.condition)
        value_true  = self._print(value_true)
        value_false = self._print(value_false)

        stmt = 'merge({true}, {false}, {cond})'.format(
                true=value_true,
                false=value_false,
                cond=condition)

        return stmt

    def _print_NumpyArray(self, expr):
        order = expr.order

        # If Numpy array is stored with column-major ordering, transpose values
        # use reshape with order for rank > 2
        if expr.rank <= 2:
            arg = self._apply_cast(expr.dtype, expr.arg)
            rhs_code = self._get_node_without_gFTL(arg)
            if expr.arg.order and expr.arg.order != expr.order:
                rhs_code = f'transpose({rhs_code})'
            if expr.arg.rank < expr.rank:
                if order == 'F':
                    shape_code = ', '.join(self._print(i) for i in expr.shape)
                else:
                    shape_code = ', '.join(self._print(i) for i in expr.shape[::-1])
                rhs_code = f"reshape({rhs_code}, [{shape_code}])"
        else:
            expr_args = (expr.arg,) if isinstance(expr.arg, Variable) else expr.arg
            expr_args = tuple(self._apply_cast(expr.dtype, a) for a in expr_args)
            new_args = []
            inv_order = 'C' if order == 'F' else 'F'
            for a in expr_args:

                # Pack list/tuple of array/list/tuple into array
                if a.order is None and a.rank > 1:
                    a = NumpyArray(a)
                ac = self._get_node_without_gFTL(a)

                # Reshape array element if out of order
                if a.order == inv_order:
                    shape = a.shape[::-1] if a.order == 'F' else a.shape
                    shape_code = ', '.join(self._print(i) for i in shape)
                    order_code = ', '.join(self._print(LiteralInteger(i)) for i in range(a.rank, 0, -1))
                    ac = f'reshape({ac}, [{shape_code}], order=[{order_code}])'
                new_args.append(ac)

            if len(new_args) == 1:
                rhs_code = new_args[0]
            else:
                rhs_code = '[' + ' ,'.join(new_args) + ']'

            if len(new_args) != 1 or expr.arg.rank < expr.rank:
                if order == 'C':
                    shape_code = ', '.join(self._print(i) for i in expr.shape[::-1])
                    rhs_code = f'reshape({rhs_code}, [{shape_code}])'
                else:
                    shape_code = ', '.join(self._print(i) for i in expr.shape)
                    order_index = [LiteralInteger(i) for i in range(1, expr.rank+1)]
                    order_index = order_index[1:]+ order_index[:1]
                    order_code = ', '.join(self._print(i) for i in order_index)
                    rhs_code = f'reshape({rhs_code}, [{shape_code}], order=[{order_code}])'


        return rhs_code

    def _print_NumpyFloor(self, expr):
        result_code = self._print_MathFloor(expr)
        return 'real({}, {})'.format(result_code, self.print_kind(expr))

    def _print_NumpyArange(self, expr):
        start  = self._print(expr.start)
        step   = self._print(expr.step)
        shape  = PyccelMinus(expr.shape[0], LiteralInteger(1), simplify = True)
        index  = self.scope.get_temporary_variable(PythonNativeInt())

        code = '[({start} + {step} * {index}, {index} = {0}, {shape}, {1})]'
        code = code.format(self._print(LiteralInteger(0)),
                           self._print(LiteralInteger(1)),
                           start  = start,
                           step   = step,
                           index  = self._print(index),
                           shape  = self._print(shape))

        return code

    def _print_NumpyMod(self, expr):
        return self._print(PyccelMod(*expr.args))

    # ======================================================================= #
    def _print_PyccelArraySize(self, expr):
        init_value = self._print(expr.arg)
        prec = self.print_kind(expr)
        return f'size({init_value}, kind={prec})'

    def _print_PyccelArrayShapeElement(self, expr):
        arg = expr.arg
        arg_code = self._print(arg)
        prec = self.print_kind(expr)

        if isinstance(arg.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            if arg.rank == 1:
                return f'size({arg_code}, kind={prec})'

            if arg.order == 'C':
                index = PyccelMinus(LiteralInteger(arg.rank), expr.index, simplify = True)
                index = self._print(index)
            else:
                index = PyccelAdd(expr.index, LiteralInteger(1), simplify = True)
                index = self._print(index)

            return f'size({arg_code}, {index}, {prec})'
        elif isinstance(arg.class_type, (HomogeneousListType, HomogeneousSetType, DictType)):
            return f'{arg_code} % size()'
        else:
            raise NotImplementedError(f"Don't know how to represent shape of object of type {arg.class_type}")

    def _print_PythonInt(self, expr):
        value = self._print(expr.arg)
        kind = self.print_kind(expr)
        if isinstance(expr.arg.dtype.primitive_type, PrimitiveBooleanType):
            code = f'MERGE(1_{kind}, 0_{kind}, {value})'
        else:
            code  = f'Int({value}, kind = {kind})'
        return code

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        kind = self.print_kind(expr)
        if isinstance(expr.arg.dtype.primitive_type, PrimitiveBooleanType):
            code = f'MERGE(1.0_{kind}, 0.0_{kind}, {value})'
        else:
            code  = f'Real({value}, kind = {kind})'
        return code

    def _print_PythonComplex(self, expr):
        kind = self.print_kind(expr)
        if expr.is_cast:
            var = expr.internal_var
            if isinstance(var.class_type.primitive_type, PrimitiveBooleanType):
                var = PythonInt(var)
            var_code = self._print(var)
            code = f'cmplx({var_code}, kind = {kind})'
        else:
            real = self._print(expr.real)
            imag = self._print(expr.imag)
            code = f'cmplx({real}, {imag}, {kind})'
        return code

    def _print_PythonBool(self, expr):
        value = self._print(expr.arg)
        kind = self.print_kind(expr)
        if isinstance(expr.arg.dtype.primitive_type, PrimitiveBooleanType):
            return f'logical({value}, kind = {kind})'
        else:
            return f'({value} /= 0)'

    def _print_MathFloor(self, expr):
        arg = expr.args[0]
        arg_code = self._print(arg)

        # math.floor on integer argument is identity,
        # but we need parentheses around expressions
        if isinstance(arg.dtype.primitive_type, PrimitiveIntegerType):
            return f'({arg_code})'

        kind = self.print_kind(expr)
        return f'floor({arg_code}, kind = {kind})'

    def _print_NumpyRand(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')

        if (not self._additional_code):
            self._additional_code = ''
        var = self.scope.get_temporary_variable(expr.dtype, memory_handling = 'stack',
                shape = expr.shape)

        self._additional_code = self._additional_code + self._print(Assign(var,expr)) + '\n'
        return self._print(var)

    def _print_NumpyRandint(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        if expr.low is None:
            randfloat = self._print(PyccelMul(expr.high, NumpyRand(), simplify = True))
        else:
            randfloat = self._print(PyccelAdd(PyccelMul(PyccelMinus(expr.high, expr.low, simplify = True), NumpyRand(), simplify=True), expr.low, simplify = True))

        prec_code = self.print_kind(expr)
        return 'floor({}, kind={})'.format(randfloat, prec_code)

    def _print_NumpyFull(self, expr):

        # Create statement for initialization
        init_value = self._print(expr.fill_value)
        return init_value
    
    def _print_NumpyAmax(self, expr):
        array_arg = expr.arg
        if isinstance(array_arg.dtype.primitive_type, PrimitiveBooleanType):
            array_arg = NumpyInt32(array_arg)
        arg_code = self._get_node_without_gFTL(array_arg)

        if isinstance(array_arg.dtype.primitive_type, PrimitiveComplexType):
            self.add_import(Import('pyc_math_f90', Module('pyc_math_f90',(),())))
            return f'amax({array_arg})'
        else:
            return f'maxval({arg_code})'
    
    def _print_NumpyAmin(self, expr):
        array_arg = expr.arg
        if isinstance(array_arg.dtype.primitive_type, PrimitiveBooleanType):
            array_arg = NumpyInt32(array_arg)
        arg_code = self._get_node_without_gFTL(array_arg)

        if isinstance(array_arg.dtype.primitive_type, PrimitiveComplexType):
            self.add_import(Import('pyc_math_f90', Module('pyc_math_f90',(),())))
            return f'amin({array_arg})'
        else:
            return f'minval({arg_code})'
        
    def _print_PythonMin(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            arg_code = self._get_node_without_gFTL(arg)
            code = f'minval({arg_code})'
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'min('+code+')'
        return code

    def _print_PythonMax(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            arg_code = self._get_node_without_gFTL(arg)
            code = f'maxval({arg_code})'
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'max('+code+')'
        return code

    # ... MACROS
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

    # ...
    def _print_MacroType(self, expr):
        dtype = self._print(expr.argument.dtype)
        prec  = expr.argument.dtype.precision

        if dtype == 'integer':
            if prec==4:
                return 'MPI_INTEGER'
            elif prec==8:
                return 'MPI_INTEGER8'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        elif dtype == 'float':
            if prec==8:
                return 'MPI_DOUBLE'
            if prec==4:
                return 'MPI_FLOAT'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                    severity='fatal')

        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

    def _print_MacroCount(self, expr):

        var = expr.argument

        if var.rank == 0:
            return '1'
        else:
            return self._print(functools.reduce(
                lambda x,y: PyccelMul(x,y,simplify=True), var.shape))

    def _print_Declare(self, expr):
        # ... ignored declarations
        var = expr.variable
        expr_type = var.class_type
        if isinstance(expr_type, SymbolicType):
            return ''

        # meta-variables
        if (isinstance(expr.variable, Variable) and
            expr.variable.name.startswith('__')):
            return ''
        # ...

        if isinstance(expr_type, InhomogeneousTupleType):
            return ''

        # ... TODO improve
        # Group the variables by intent
        dtype           = var.dtype
        rank            = var.rank
        shape           = var.alloc_shape
        is_const        = var.is_const
        is_optional     = var.is_optional
        is_private      = var.is_private
        is_alias        = var.is_alias and not isinstance(dtype, BindCPointer)
        on_heap         = var.on_heap
        on_stack        = var.on_stack
        is_static       = expr.static
        is_external     = expr.external
        is_target       = var.is_target and not var.is_alias
        intent          = expr.intent
        intent_in = intent and intent != 'out'
        # ...

        dtype_str = ''
        rankstr   = ''

        # ... print datatype
        if isinstance(expr_type, CustomDataType):
            name   = expr_type.name

            if var.is_argument:
                sig = 'class'
            else:
                sig = 'type'
            dtype_str = f'{sig}({name})'
        elif isinstance(dtype, FixedSizeNumericType) and \
                isinstance(expr_type, (NumpyNDArrayType, HomogeneousTupleType, FixedSizeNumericType)):
            dtype_str = self._print(dtype.primitive_type)
            dtype_str += f'({self.print_kind(var)})'

            if rank > 0:
                # arrays are 0-based in pyccel, to avoid ambiguity with range
                start_val = self._print(LiteralInteger(0))

                if intent_in:
                    rankstr = ', '.join([f'{start_val}:'] * rank)
                elif is_static or on_stack:
                    ordered_shape = shape[::-1] if var.order == 'C' else shape
                    ubounds = [PyccelMinus(s, LiteralInteger(1), simplify = True) for s in ordered_shape]
                    rankstr = ', '.join(f'{start_val}:{self._print(u)}' for u in ubounds)
                elif is_alias or on_heap:
                    rankstr = ', '.join(':'*rank)
                else:
                    raise NotImplementedError("Fortran rank string undetermined")
                rankstr = f'({rankstr})'

        elif isinstance(expr_type, (HomogeneousListType, HomogeneousSetType, DictType)):
            self.add_import(self._build_gFTL_module(expr_type))
            typename = self._print(expr_type)
            dtype_str = f'type({typename})'
        elif isinstance(dtype, StringType):
            dtype_str = self._print(dtype)

            if intent_in:
                dtype_str = dtype_str[:9] +'(len =*)'
                #TODO improve ,this is the case of character as argument
        elif isinstance(dtype, BindCPointer):
            dtype_str = 'type(c_ptr)'
            self._constantImports.setdefault('ISO_C_Binding', set()).add('c_ptr')
        else:
            errors.report(f"Don't know how to print type {expr_type} in Fortran",
                    symbol=expr, severity='fatal')

        code_value = ''
        if expr.value:
            code_value = ' = {0}'.format(self._print(expr.value))

        vstr = self._print(expr.variable.name)

        # Default empty strings
        intentstr      = ''
        allocatablestr = ''
        optionalstr    = ''
        privatestr     = ''
        externalstr    = ''
        is_string = isinstance(var.class_type, StringType)

        # Compute intent string
        if intent:
            if intent == 'in' and rank == 0 and not is_optional \
                and not isinstance(expr_type, CustomDataType):
                intentstr = ', value'
                if is_const:
                    intentstr += ', intent(in)'
            else:
                intentstr = f', intent({intent})'

        # Compute allocatable string
        if not is_static and not is_string:
            if is_alias:
                allocatablestr = ', pointer'

            elif on_heap and not intent_in and isinstance(var.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
                allocatablestr = ', allocatable'

            # ISSUES #177: var is allocatable and target
            if is_target:
                allocatablestr = f'{allocatablestr}, target'

        # Compute optional string
        if is_optional:
            optionalstr = ', optional'

        # Compute private string
        if is_private:
            privatestr = ', private'

        # Compute external string
        if is_external:
            externalstr = ', external'

        mod_str = ''
        if expr.module_variable and not is_private and isinstance(expr.variable.class_type, FixedSizeNumericType):
            mod_str = ', bind(c)'

        # Construct declaration
        left  = dtype_str + allocatablestr + optionalstr + privatestr + externalstr + mod_str + intentstr
        right = vstr + rankstr + code_value
        return f'{left} :: {right}\n'

    def _print_AliasAssign(self, expr):
        code = ''
        lhs = expr.lhs
        rhs = expr.rhs

        if isinstance(lhs.class_type, InhomogeneousTupleType):
            return self._print(CodeBlock([AliasAssign(l, r) for l,r in zip(lhs,rhs)]))
        if isinstance(rhs, FunctionCall):
            return self._print(rhs)

        # TODO improve
        op = '=>'
        shape_code = ''
        if lhs.rank > 0:
            shape_code = ', '.join('0:' for i in range(lhs.rank))
            shape_code = '({s_c})'.format(s_c = shape_code)

        code += '{lhs}{s_c} {op} {rhs}'.format(lhs=self._print(expr.lhs),
                                          s_c = shape_code,
                                          op=op,
                                          rhs=self._print(expr.rhs))

        return code + '\n'

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs = expand_to_loops(expr,
                    self.scope.get_temporary_variable, self.scope,
                    language_has_vectors = True)
        else:
            body_exprs = expr.body
        body_stmts = []
        for b in body_exprs :
            line = self._print(b)
            if (self._additional_code):
                body_stmts.append(self._additional_code)
                self._additional_code = None
            body_stmts.append(line)
        return ''.join(self._print(b) for b in body_stmts)

    # TODO the ifs as they are are, is not optimal => use elif
    def _print_SymbolicAssign(self, expr):
        errors.report(FOUND_SYMBOLIC_ASSIGN,
                      symbol=expr.lhs, severity='warning')

        stmt = Comment(str(expr))
        return self._print_Comment(stmt)

    def _print_NumpyReal(self, expr):
        value = self._print(expr.internal_var)
        code = 'Real({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_Assign(self, expr):
        rhs = expr.rhs

        if isinstance(rhs, FunctionCall):
            return self._print(rhs)

        lhs_code = self._print(expr.lhs)
        # we don't print Range
        # TODO treat the case of iterable classes
        if isinstance(rhs, PyccelUnarySub) and rhs.args[0] == INF:
            rhs_code = '-Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if rhs == INF:
            rhs_code = 'Huge({0})'.format(lhs_code)
            return '{0} = {1}\n'.format(lhs_code, rhs_code)

        if isinstance(rhs, (PythonRange, Product)):
            return ''

        if isinstance(rhs, NumpyRand):
            return 'call random_number({0})\n'.format(self._print(expr.lhs))

        if isinstance(rhs, NumpyEmpty):
            return ''

        if isinstance(rhs, NumpyNonZero):
            code = ''
            lhs = expr.lhs
            for i,e in enumerate(rhs.elements):
                l_c = self._print(lhs[i])
                e_c = self._print(e)
                code += '{0} = {1}\n'.format(l_c,e_c)
            return code

        if isinstance(rhs, ConstructorCall):
            func = rhs.func
            name = str(func.name)

            # TODO uncomment later

#            # we don't print the constructor call if iterable object
#            if this.dtype.is_iterable:
#                return ''
#
#            # we don't print the constructor call if with construct object
#            if this.dtype.is_with_construct:
#                return ''

            if name == "__init__":
                name = "create"
            rhs_code = self._print(name)
            rhs_code = '{0} % {1}'.format(lhs_code, rhs_code)

            code_args = ', '.join(self._print(i) for i in rhs.arguments)
            return 'call {0}({1})\n'.format(rhs_code, code_args)

        if (isinstance(expr.lhs, Variable) and
              isinstance(expr.lhs.dtype, SymbolicType)):
            return ''

        # Right-hand side code
        rhs_code = self._print(rhs)

        code = ''
        # if (expr.status == 'unallocated') and not (expr.like is None):
        #     stmt = ZerosLike(lhs=lhs_code, rhs=expr.like)
        #     code += self._print(stmt)
        #     code += '\n'
        code += '{0} = {1}'.format(lhs_code, rhs_code)
#        else:
#            code_args = ''
#            func = expr.rhs
#            # func here is of instance FunctionCall
#            cls_name = func.func.cls_name
#            keys = func.func.arguments

#            # for MPI statements, we need to add the lhs as the last argument
#            # TODO improve
#            if isinstance(func.func, MPI):
#                if not func.arguments:
#                    code_args = lhs_code
#                else:
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#            else:
#                _ij_print = lambda i, j: '{0}={1}'.format(self._print(i), \
#                                                         self._print(j))
#
#                code_args = ', '.join(_ij_print(i, j) \
#                                      for i, j in zip(keys, func.arguments))
#            if (not func.arguments is None) and (len(func.arguments) > 0):
#                if (not cls_name):
#                    code_args = ', '.join(self._print(i) for i in func.arguments)
#                    code_args = '{0}, {1}'.format(code_args, lhs_code)
#                else:
#            print('code_args > {0}'.format(code_args))
#            code = 'call {0}({1})'.format(rhs_code, code_args)
        return code + '\n'

#------------------------------------------------------------------------------
    def _print_Allocate(self, expr):
        class_type = expr.variable.class_type
        if isinstance(class_type, (NumpyNDArrayType, HomogeneousTupleType, CustomDataType)):
            # Transpose indices because of Fortran column-major ordering
            shape = expr.shape if expr.order == 'F' else expr.shape[::-1]

            var_code = self._print(expr.variable)
            size_code = ', '.join(self._print(i) for i in shape)
            shape_code = ', '.join('0:' + self._print(PyccelMinus(i, LiteralInteger(1), simplify = True)) for i in shape)
            if shape:
                shape_code = f'({shape_code})'
            code = ''

            if expr.status == 'unallocated':
                code += f'allocate({var_code}{shape_code})\n'

            elif expr.status == 'unknown':
                code += f'if (allocated({var_code})) then\n'
                code += f'  if (any(size({var_code}) /= [{size_code}])) then\n'
                code += f'    deallocate({var_code})\n'
                code += f'    allocate({var_code}{shape_code})\n'
                code +=  '  end if\n'
                code +=  'else\n'
                code += f'  allocate({var_code}{shape_code})\n'
                code +=  'end if\n'

            elif expr.status == 'allocated':
                code += f'if (any(size({var_code}) /= [{size_code}])) then\n'
                code += f'  deallocate({var_code})\n'
                code += f'  allocate({var_code}{shape_code})\n'
                code +=  'end if\n'

            return code

        elif isinstance(class_type, (HomogeneousContainerType, DictType)):
            return ''

        else:
            return self._print_not_supported(expr)

#-----------------------------------------------------------------------------
    def _print_Deallocate(self, expr):
        var = expr.variable
        class_type = var.class_type
        if isinstance(class_type, InhomogeneousTupleType):
            return ''.join(self._print(Deallocate(v)) for v in var)

        if isinstance(class_type, CustomDataType):
            Pyccel__del = expr.variable.cls_base.scope.find('__del__')
            Pyccel_del_args = [FunctionCallArgument(var)]
            return self._print(FunctionCall(Pyccel__del, Pyccel_del_args))

        if var.is_alias or isinstance(class_type, (HomogeneousListType, HomogeneousSetType, DictType)):
            return ''
        elif isinstance(class_type, (NumpyNDArrayType, HomogeneousTupleType, StringType)):
            var_code = self._print(var)
            code  = 'if (allocated({})) then\n'.format(var_code)
            code += '  deallocate({})\n'     .format(var_code)
            code += 'end if\n'
            return code
        else:
            errors.report(f"Deallocate not implemented for {class_type}",
                    severity='error', symbol=expr)
            return ''

    def _print_DeallocatePointer(self, expr):
        var_code = self._print(expr.variable)
        return f'deallocate({var_code})'

#------------------------------------------------------------------------------

    def _print_PrimitiveBooleanType(self, expr):
        return 'logical'

    def _print_PrimitiveIntegerType(self, expr):
        return 'integer'

    def _print_PrimitiveFloatingPointType(self, expr):
        return 'real'

    def _print_PrimitiveComplexType(self, expr):
        return 'complex'

    def _print_StringType(self, expr):
        return 'character(len=280)'
        #TODO fix improve later

    def _print_FixedSizeNumericType(self, expr):
        return f'{self._print(expr.primitive_type)}{expr.precision}'

    def _print_PythonNativeBool(self, expr):
        return 'logical'

    def _print_HomogeneousListType(self, expr):
        return 'Vector_'+self._print(expr.element_type)

    def _print_HomogeneousSetType(self, expr):
        return 'Set_'+self._print(expr.element_type)

    def _print_PairType(self, expr):
        return 'Pair_'+self._print(expr.key_type)+'__'+self._print(expr.value_type)

    def _print_DictType(self, expr):
        return 'Map_'+self._print(expr.key_type)+'__'+self._print(expr.value_type)

    def _print_IteratorType(self, expr):
        iterable_type = self._print(expr.iterable_type)
        return f"{iterable_type}_Iterator"

    def _print_DataType(self, expr):
        return self._print(expr.name)

    def _print_LiteralString(self, expr):
        sp_chars = ['\a', '\b', '\f', '\r', '\t', '\v', "'", '\n']
        sub_str = ''
        formatted_str = []
        for c in expr.python_value:
            if c in sp_chars:
                if sub_str != '':
                    formatted_str.append("'{}'".format(sub_str))
                    sub_str = ''
                formatted_str.append('ACHAR({})'.format(ord(c)))
            else:
                sub_str += c
        if sub_str != '':
            formatted_str.append("'{}'".format(sub_str))
        return ' // '.join(formatted_str)

    def _print_Interface(self, expr):
        interface_funcs = expr.functions

        example_func = interface_funcs[0]

        # ... we don't print 'hidden' functions
        if example_func.is_inline:
            return ''

        if len(example_func.results) == 1:
            if len(set(f.results[0].var.rank == 0 for f in interface_funcs)) != 1:
                message = ("Fortran cannot yet handle a templated function returning either a scalar or an array. "
                           "If you are using the terminal interface, please pass --language c, "
                           "if you are using the interactive interfaces epyccel or lambdify, please pass language='c'. "
                           "See https://github.com/pyccel/pyccel/issues/1339 to monitor the advancement of this issue.")
                errors.report(message,
                        severity='error', symbol=expr)

        name = self._print(expr.name)
        if all(isinstance(f, FunctionAddress) for f in interface_funcs):
            funcs = interface_funcs
        else:
            funcs = [f for f in interface_funcs if f is \
                    expr.point([FunctionCallArgument(a.var.clone('arg_'+str(i))) \
                        for i,a in enumerate(f.arguments)])]

        if expr.is_argument:
            funcs_sigs = []
            for f in funcs:
                parts = self.function_signature(f, f.name)
                parts = ["{}({}) {}\n".format(parts['sig'], parts['arg_code'], parts['func_end']),
                        self.print_constant_imports()+'\n',
                        parts['arg_decs'],
                        'end {} {}\n'.format(parts['func_type'], f.name)]
                funcs_sigs.append(''.join(a for a in parts))
            interface = 'interface\n' + '\n'.join(a for a in funcs_sigs) + 'end interface\n'
            return interface

        if funcs[0].cls_name:
            for k, m in list(_default_methods.items()):
                name = name.replace(k, m)
            cls_name = expr.cls_name
            if not (cls_name == '__UNDEFINED__'):
                name = '{0}_{1}'.format(cls_name, name)
        else:
            for i in _default_methods:
                # because we may have a class Point with init: Point___init__
                if i in name:
                    name = name.replace(i, _default_methods[i])
        interface = 'interface ' + name +'\n'
        for f in funcs:
            interface += 'module procedure ' + str(f.name)+'\n'
        interface += 'end interface\n'
        return interface



   # def _print_With(self, expr):
   #     self.set_scope(expr)
   #     test = 'call '+self._print(expr.test) + '%__enter__()'
   #     body = self._print(expr.body)
   #     end = 'call '+self._print(expr.test) + '%__exit__()'
   #     code = ('{test}\n'
   #            '{body}\n'
   #            '{end}').format(test=test, body=body, end=end)
        #TODO return code later
  #      expr.block
  #      self.exit_scope()
  #      return ''

    def _print_FunctionAddress(self, expr):
        return expr.name

    def function_signature(self, expr, name):
        """
        Get the different parts of the signature of the function `expr`.

        A helper function to print just the signature of the function
        including the declarations of the arguments and results.

        Parameters
        ----------
        expr : FunctionDef
            The function whose signature should be printed.
        name : str
            The name which should be printed as the name of the function.
            (May be different from expr.name in the case of interfaces).

        Returns
        -------
        dict
            A dictionary with the keys :
                sig - The declaration of the function/subroutine with any necessary keywords.
                arg_code - A string containing a list of the arguments.
                func_end - Any code to be added to the signature after the arguments (ie result).
                arg_decs - The code necessary to declare the arguments of the function/subroutine.
                func_type - Subroutine or function.
        """
        is_pure      = expr.is_pure
        is_elemental = expr.is_elemental
        out_args = [r.var for r in expr.results if not r.is_argument]
        args_decs = OrderedDict()
        arguments = expr.arguments
        argument_vars = [a.var for a in arguments]

        func_end  = ''
        rec = 'recursive ' if expr.is_recursive else ''
        if len(out_args) != 1 or out_args[0].rank > 0:
            func_type = 'subroutine'
            for result in out_args:
                args_decs[result] = Declare(result, intent='out')

            functions = expr.functions

        else:
           #todo: if return is a function
            func_type = 'function'
            result = out_args[0]
            functions = expr.functions

            func_end = 'result({0})'.format(result.name)

            args_decs[result] = Declare(result)
            out_args = []
        # ...

        for i, arg in enumerate(arguments):
            arg_var = arg.var
            if isinstance(arg_var, Variable):
                if isinstance(arg, BindCFunctionDefArgument) and arg.original_function_argument_variable.rank!=0:
                    for b_arg in arg.get_all_function_def_arguments():
                        v = b_arg.var
                        dec = Declare(v, intent='in')
                        args_decs[v] = dec
                else:
                    if i == 0 and expr.cls_name:
                        dec = Declare(arg_var, intent='inout')
                    elif arg.inout:
                        dec = Declare(arg_var, intent='inout')
                    else:
                        dec = Declare(arg_var, intent='in')
                    args_decs[arg_var] = dec

        # treat case of pure function
        sig = '{0}{1} {2}'.format(rec, func_type, name)
        if is_pure:
            sig = 'pure {}'.format(sig)

        # treat case of elemental function
        if is_elemental:
            sig = 'elemental {}'.format(sig)

        arg_code  = ', '.join(self._print(i) for i in chain( arguments, out_args ))

        arg_decs = ''.join(self._print(i) for i in args_decs.values())

        parts = {
                'sig' : sig,
                'arg_code' : arg_code,
                'func_end' : func_end,
                'arg_decs' : arg_decs,
                'func_type' : func_type
        }
        return parts

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''
        self.set_scope(expr.scope)


        name = expr.cls_name or expr.name

        sig_parts = self.function_signature(expr, name)
        bind_c = ' bind(c)' if isinstance(expr, BindCFunctionDef) else ''
        prelude = sig_parts.pop('arg_decs')
        functions = [f for f in expr.functions if not f.is_inline]
        func_interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        body_code = self._print(expr.body)
        docstring = self._print(expr.docstring) if expr.docstring else ''

        decs = [Declare(v) for v in expr.local_vars]
        self._get_external_declarations(decs)

        prelude += ''.join(self._print(i) for i in decs)
        if len(functions)>0:
            functions_code = '\n'.join(self._print(i) for  i in functions)
            body_code = body_code +'\ncontains\n' + functions_code

        imports = ''.join(self._print(i) for i in expr.imports)

        parts = [docstring,
                f"{sig_parts['sig']}({sig_parts['arg_code']}){bind_c} {sig_parts['func_end']}\n",
                imports,
                'implicit none\n',
                prelude,
                func_interfaces,
                body_code,
                'end {} {}\n'.format(sig_parts['func_type'], name)]

        self.exit_scope()

        return '\n'.join(a for a in parts if a)

    def _print_Pass(self, expr):
        return '! pass\n'

    def _print_Nil(self, expr):
        return ''

    def _print_NilArgument(self, expr):
        raise errors.report("Trying to use optional argument in inline function without providing a variable",
                symbol=expr,
                severity='fatal')

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)
        code +='return\n'
        return code

    def _print_Del(self, expr):
        return ''.join(self._print(var) for var in expr.variables)

    def _print_ClassDef(self, expr):
        # ... we don't print 'hidden' classes
        if expr.hide:
            return '', ''
        # ...
        self.set_scope(expr.scope)

        name = self._print(expr.name)
        self.set_current_class(name)
        base = None # TODO: add base in ClassDef

        decs = ''.join(self._print(Declare(i)) for i in expr.attributes)

        aliases = []
        names   = []
        methods = ''.join(f'procedure :: {method.name} => {method.cls_name}\n' for method in expr.methods)
        for i in expr.interfaces:
            names = ','.join(f.cls_name for f in i.functions if not f.is_inline)
            if names:
                methods += f'generic, public :: {i.name} => {names}\n'
                methods += f'procedure :: {names}\n'



        options = ', '.join(i for i in expr.options)

        sig = 'type, {0}'.format(options)
        if not(base is None):
            sig = '{0}, extends({1})'.format(sig, base)

        docstring = self._print(expr.docstring) if expr.docstring else ''
        code = f'{sig} :: {name}\n{decs}\n'
        code = code + 'contains\n' + methods
        decs = ''.join([docstring, code, f'end type {name}\n'])

        sep = self._print(SeparatorComment(40))
        # we rename all methods because of the aliasing
        cls_methods = [i.clone('{0}'.format(i.name)) for i in expr.methods]
        for i in expr.interfaces:
            cls_methods +=  [j.clone('{0}'.format(j.name)) for j in i.functions]

        methods = ''.join('\n'.join(['', sep, self._print(i), sep, '']) for i in cls_methods)

        self.set_current_class(None)

        self.exit_scope()

        return decs, methods

    def _print_Break(self, expr):
        return 'exit\n'

    def _print_Continue(self, expr):
        return 'cycle\n'

    def _print_AugAssign(self, expr):
        new_expr = expr.to_basic_assign()
        expr.invalidate_node()
        return self._print(new_expr)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)

        test_step = expr.step
        if isinstance(test_step, LiteralInteger) and test_step.python_value == 1:
            step = ''
        else:
            step = ', '+self._print(expr.step)

        if isinstance(test_step, PyccelUnarySub):
            test_step = expr.step.args[0]

        # testing if the step is a value or an expression
        if isinstance(test_step, Literal):
            if isinstance(expr.step, PyccelUnarySub):
                stop = PyccelAdd(expr.stop, LiteralInteger(1), simplify = True)
            else:
                stop = PyccelMinus(expr.stop, LiteralInteger(1), simplify = True)
        else:
            stop = IfTernaryOperator(PyccelGt(expr.step, LiteralInteger(0)),
                                     PyccelMinus(expr.stop, LiteralInteger(1), simplify = True),
                                     PyccelAdd(expr.stop, LiteralInteger(1), simplify = True))

        stop = self._print(stop)
        return f'{start}, {stop}{step}'

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_For(self, expr):
        self.set_scope(expr.scope)

        indices = expr.iterable.loop_counters
        index = indices[0] if indices else expr.target
        if expr.iterable.num_loop_counters_required:
            self.scope.insert_variable(index)

        target   = index
        my_range = expr.iterable.get_range()

        if not isinstance(my_range, PythonRange):
            # Only iterable currently supported is PythonRange
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        tar        = self._print(target)
        range_code = self._print(my_range)

        prolog = 'do {0} = {1}\n'.format(tar, range_code)
        epilog = 'end do\n'

        additional_assign = CodeBlock(expr.iterable.get_assigns(expr.target))
        prolog += self._print(additional_assign)

        body = self._print(expr.body)

        if expr.end_annotation:
            end_annotation = expr.end_annotation.replace("for", "do")
            epilog += end_annotation

        self.exit_scope()

        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)

    # .....................................................
    #               Print OpenMP AnnotatedComment
    # .....................................................

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            combined = expr.combined.replace("for", "do")
            clauses = ' ' + combined

        omp_expr = '!$omp {}'.format(expr.name.replace("for", "do"))
        clauses += str(expr.txt).replace("cancel for", "cancel do")
        omp_expr = '{}{}\n'.format(omp_expr, clauses)
        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        omp_expr = str(expr.txt)
        if "section" in omp_expr and "sections" not in omp_expr:
            return ''
        omp_expr = omp_expr.replace("for", "do")
        if expr.has_nowait:
            omp_expr += ' nowait'
        omp_expr = '!$omp {}\n'.format(omp_expr)
        return omp_expr
    # .....................................................

    # .....................................................
    #                   OpenACC statements
    # .....................................................
    def _print_ACC_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = ''.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = f'!$acc parallel {clauses}\n'
        epilog = '!$acc end parallel\n'
        # ...

        # ...
        code = (f'{prolog}'
                f'{body}'
                f'{epilog}')
        # ...

        return code

    def _print_ACC_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        # ...

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = f'!$acc loop {clauses}\n'
        epilog = '!$acc end loop\n'
        # ...

        # ...
        code = (f'{prolog}'
                f'{loop}'
                f'{epilog}')
        # ...

        return code

    def _print_ACC_Async(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'async({})'.format(args)

    def _print_ACC_Auto(self, expr):
        return 'auto'

    def _print_ACC_Bind(self, expr):
        return 'bind({})'.format(self._print(expr.variable))

    def _print_ACC_Collapse(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_loops))

    def _print_ACC_Copy(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copy({})'.format(args)

    def _print_ACC_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_ACC_Copyout(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyout({})'.format(args)

    def _print_ACC_Create(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'create({})'.format(args)

    def _print_ACC_Default(self, expr):
        return 'default({})'.format(self._print(expr.status))

    def _print_ACC_DefaultAsync(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'default_async({})'.format(args)

    def _print_ACC_Delete(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'delete({})'.format(args)

    def _print_ACC_Device(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device({})'.format(args)

    def _print_ACC_DeviceNum(self, expr):
        return 'collapse({0})'.format(self._print(expr.n_device))

    def _print_ACC_DevicePtr(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'deviceptr({})'.format(args)

    def _print_ACC_DeviceResident(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_resident({})'.format(args)

    def _print_ACC_DeviceType(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'device_type({})'.format(args)

    def _print_ACC_Finalize(self, expr):
        return 'finalize'

    def _print_ACC_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_ACC_Gang(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'gang({})'.format(args)

    def _print_ACC_Host(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'host({})'.format(args)

    def _print_ACC_If(self, expr):
        return 'if({})'.format(self._print(expr.test))

    def _print_ACC_Independent(self, expr):
        return 'independent'

    def _print_ACC_Link(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'link({})'.format(args)

    def _print_ACC_NoHost(self, expr):
        return 'nohost'

    def _print_ACC_NumGangs(self, expr):
        return 'num_gangs({0})'.format(self._print(expr.n_gang))

    def _print_ACC_NumWorkers(self, expr):
        return 'num_workers({0})'.format(self._print(expr.n_worker))

    def _print_ACC_Present(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'present({})'.format(args)

    def _print_ACC_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_ACC_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_ACC_Self(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'self({})'.format(args)

    def _print_ACC_Seq(self, expr):
        return 'seq'

    def _print_ACC_Tile(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'tile({})'.format(args)

    def _print_ACC_UseDevice(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'use_device({})'.format(args)

    def _print_ACC_Vector(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector({})'.format(args)

    def _print_ACC_VectorLength(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'vector_length({})'.format(self._print(expr.n))

    def _print_ACC_Wait(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'wait({})'.format(args)

    def _print_ACC_Worker(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'worker({})'.format(args)
    # .....................................................

    #def _print_Block(self, expr):
    #    body    = '\n'.join(self._print(i) for i in expr.body)
    #    prelude = '\n'.join(self._print(i) for i in expr.declarations)
    #    return prelude, body

    def _print_While(self,expr):
        self.set_scope(expr.scope)
        body = self._print(expr.body)
        self.exit_scope()
        return ('do while ({test})\n'
                '{body}'
                'end do\n').format(test=self._print(expr.test), body=body)

    def _print_ErrorExit(self, expr):
        # TODO treat the case of MPI
        return 'STOP'

    def _print_Assert(self, expr):
        prolog = "if ( .not. ({0})) then".format(self._print(expr.test))
        body = 'stop 1'
        epilog = 'end if'
        return ('{prolog}\n'
                '{body}\n'
                '{epilog}\n').format(prolog=prolog, body=body, epilog=epilog)

    def _handle_not_none(self, lhs, lhs_var):
        """
        Print code for `x is not None` statement.

        Print the code which checks if x is not None. This means different
        things depending on the type of `x`. If `x` is optional it checks
        if it is present, if `x` is a C pointer it checks if it points at
        anything.

        Parameters
        ----------
        lhs : str
            The code representing `x`.
        lhs_var : Variable
            The Variable `x`.

        Returns
        -------
        str
            The code which checks if `x is not None`.
        """
        if isinstance(lhs_var.dtype, BindCPointer):
            self._constantImports.setdefault('ISO_C_Binding', set()).add('c_associated')
            return f'c_associated({lhs})'
        else:
            return f'present({lhs})'

    def _print_PyccelIs(self, expr):
        lhs_var = expr.lhs
        rhs_var = expr.rhs
        lhs = self._print(lhs_var)
        rhs = self._print(rhs_var)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(rhs_var, Nil):
            return '.not. '+ self._handle_not_none(lhs, lhs_var)

        if all(isinstance(var.dtype.primitive_type, PrimitiveBooleanType) for var in (a, b)):
            return f'{lhs} .eqv. {rhs}'

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_PyccelIsNot(self, expr):
        lhs_var = expr.lhs
        rhs_var = expr.rhs
        lhs = self._print(lhs_var)
        rhs = self._print(rhs_var)
        a = expr.args[0]
        b = expr.args[1]

        if isinstance(rhs_var, Nil):
            return self._handle_not_none(lhs, lhs_var)

        if all(isinstance(var.dtype.primitive_type, PrimitiveBooleanType) for var in (a, b)):
            return f'{lhs} .neqv. {rhs}'

        errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                      symbol=expr, severity='fatal')

    def _print_If(self, expr):
        # ...

        lines = []

        for i, (c, e) in enumerate(expr.blocks):

            if i == 0:
                lines.append("if (%s) then\n" % self._print(c))
            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else\n")
            else:
                lines.append("else if (%s) then\n" % self._print(c))

            if isinstance(e, (list, tuple, PythonTuple)):
                lines.extend(self._print(ee) for ee in e)
            else:
                lines.append(self._print(e))

        lines.append("end if\n")

        return ''.join(lines)

    def _print_IfTernaryOperator(self, expr):

        cond = PythonBool(expr.cond) if not isinstance(expr.cond.dtype.primitive_type, PrimitiveBooleanType) else expr.cond
        value_true, value_false = self._apply_cast(expr.dtype, expr.value_true, expr.value_false)

        cond = self._print(cond)
        value_true = self._print(value_true)
        value_false = self._print(value_false)
        return 'merge({true}, {false}, {cond})'.format(cond = cond, true = value_true, false = value_false)

    def _print_PyccelPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_PyccelAdd(self, expr):
        if isinstance(expr.dtype, StringType):
            return '//'.join('trim('+self._print(a)+')' for a in expr.args)
        else:
            args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
            return ' + '.join(self._print(a) for a in args)

    def _print_PyccelMinus(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        args_code = [self._print(a) for a in args]

        return ' - '.join(args_code)

    def _print_PyccelMul(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        args_code = [self._print(a) for a in args]
        return ' * '.join(a for a in args_code)

    def _print_PyccelDiv(self, expr):
        if all(isinstance(a.dtype.primitive_type, (PrimitiveBooleanType, PrimitiveIntegerType)) for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return ' / '.join(self._print(a) for a in args)

    def _print_PyccelMod(self, expr):
        is_float = isinstance(expr.dtype.primitive_type, PrimitiveFloatingPointType)

        def correct_type_arg(a):
            if is_float and isinstance(a.dtype.primitive_type, PrimitiveIntegerType):
                return NumpyFloat(a)
            else:
                return a

        args = [self._print(correct_type_arg(a)) for a in expr.args]

        code = args[0]
        for c in args[1:]:
            code = 'MODULO({},{})'.format(code, c)
        return code

    def _print_PyccelFloorDiv(self, expr):

        code     = self._print(expr.args[0])
        adtype   = expr.args[0].dtype.primitive_type
        is_float = isinstance(expr.dtype.primitive_type, PrimitiveFloatingPointType)
        for b in expr.args[1:]:
            bdtype    = b.dtype.primitive_type
            if all(isinstance(dtype, PrimitiveIntegerType) for dtype in (adtype, bdtype)):
                b = NumpyFloat(b)
            c = self._print(b)
            adtype = bdtype
            code = 'FLOOR({}/{},{})'.format(code, c, self.print_kind(expr))
            if is_float:
                code = 'real({}, {})'.format(code, self.print_kind(expr))
        return code

    def _print_PyccelRShift(self, expr):
        return 'RSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return 'LSHIFT({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        if isinstance(expr.dtype.primitive_type, PrimitiveBooleanType):
            return ' .neqv. '.join(self._print(a) for a in expr.args)
        return 'IEOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        if isinstance(expr.dtype.primitive_type, PrimitiveBooleanType):
            return ' .or. '.join(self._print(a) for a in expr.args)
        return 'IOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        if isinstance(expr.dtype.primitive_type, PrimitiveBooleanType):
            return ' .and. '.join(self._print(a) for a in expr.args)
        return 'IAND({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelInvert(self, expr):
        return 'NOT({})'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        args = [a if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else PythonBool(a) for a in expr.args]
        return ' .and. '.join(self._print(a) for a in args)

    def _print_PyccelOr(self, expr):
        args = [a if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else PythonBool(a) for a in expr.args]
        return ' .or. '.join(self._print(a) for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype.primitive_type
        b = expr.args[1].dtype.primitive_type

        if all(isinstance(var, PrimitiveBooleanType) for var in (a, b)):
            return '{} .eqv. {}'.format(lhs, rhs)
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype.primitive_type
        b = expr.args[1].dtype.primitive_type

        if all(isinstance(var, PrimitiveBooleanType) for var in (a, b)):
            return '{} .neqv. {}'.format(lhs, rhs)
        return '{0} /= {1}'.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        args = [PythonInt(a) if isinstance(a.dtype.primitive_type, PrimitiveBooleanType) else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        if not isinstance(expr.args[0].dtype.primitive_type, PrimitiveBooleanType):
            return '{} == 0'.format(a)
        return '.not. {}'.format(a)

    def _print_Header(self, expr):
        return ''

    def _print_SysExit(self, expr):
        code = ""
        exit_code = expr.status
        if isinstance(exit_code, LiteralInteger):
            arg = exit_code.python_value
        elif not isinstance(getattr(exit_code.dtype, 'primitive_type'), PrimitiveIntegerType) or exit_code.rank > 0:
            print_arg = FunctionCallArgument(exit_code)
            code = self._print(PythonPrint((print_arg, ), file="stderr"))
            arg = "1"
        else:
            if exit_code.dtype.precision != 4:
                exit_code = NumpyInt32(exit_code)
            arg = self._print(exit_code)
        return f'{code}stop {arg}\n'

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_fortran[type_name]
        except KeyError:
            self._print_not_supported(expr)
        if func_name.startswith('ieee_'):
            self._constantImports.setdefault('ieee_arithmetic', set()).add(func_name)
        args = [self._get_node_without_gFTL(NumpyFloat(a) if isinstance(a.dtype.primitive_type, PrimitiveIntegerType) else a)\
				for a in expr.args]
        code_args = ', '.join(args)
        code = f'{func_name}({code_args})'
        return code

    def _print_NumpyIsInf(self, expr):
        code = PyccelAssociativeParenthesis(PyccelAnd(
                    PyccelNot(NumpyIsFinite(expr.arg)),
                    PyccelNot(NumpyIsNan(expr.arg))))
        return self._print(code)

    def _print_NumpySign(self, expr):
        """ Print the corresponding Fortran function for a call to Numpy.sign

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with Numpy.sign call

        Returns
        -------
            string
                Equivalent internal function in Fortran

        Example
        -------
            import numpy

            numpy.sign(x) => numpy_sign(x)
            numpy_sign is an interface which calls the proper function depending on the data type of x

        """
        arg = expr.args[0]
        arg_code = self._print(arg)
        if isinstance(expr.dtype.primitive_type, PrimitiveComplexType):
            func_name = 'csgn' if numpy_v1 else 'csign'
            func = PyccelFunctionDef(func_name, NumpySign)
            self.add_import(Import('pyc_math_f90', AsName(func, func_name)))
            return f'{func_name}({arg_code})'
        else:
            # The absolute value of the result (0 if the argument is 0, 1 otherwise)
            abs_result = self._print(self._apply_cast(expr.dtype, PythonBool(arg)))
            return f'sign({abs_result}, {arg_code})'

    def _print_NumpyTranspose(self, expr):
        var = expr.internal_var
        arg = self._print(var)
        assigns = expr.get_user_nodes(Assign)
        if assigns and assigns[0].lhs.order != var.order:
            return arg
        elif var.rank == 2:
            return 'transpose({0})'.format(arg)
        else:
            var_shape = var.shape[::-1] if var.order == 'F' else var.shape
            shape = ', '.join(self._print(i) for i in var_shape)
            order = ', '.join(self._print(LiteralInteger(i)) for i in range(var.rank, 0, -1))
            return 'reshape({}, shape=[{}], order=[{}])'.format(arg, shape, order)

    def _print_MathFunctionBase(self, expr):
        """ Convert a Python expression with a math function call to Fortran
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math function call

        Returns
        -------
            string
                Equivalent expression in Fortran language

        ------
        Example:
        --------
            math.cos(x)    ==> cos(x)
            math.gcd(x, y) ==> pyc_gcd(x, y) # with include of pyc_math module
        """
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_fortran[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        if func_name.startswith("pyc"):
            self.add_import(Import('pyc_math_f90', Module('pyc_math_f90',(),())))
        args = []
        for arg in expr.args:
            if arg.dtype != expr.dtype:
                args.append(self._print(self._apply_cast(expr.dtype, arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathCeil(self, expr):
        """Convert a Python expression with a math ceil function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if isinstance(arg.dtype.primitive_type, PrimitiveIntegerType):
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "ceiling({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if isinstance(arg.dtype.primitive_type, PrimitiveIntegerType):
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if isinstance(arg.dtype.primitive_type, PrimitiveIntegerType):
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "dint({})".format(code_arg)

    def _print_MathPow(self, expr):
        base = expr.args[0]
        e    = expr.args[1]

        base_c = self._print(base)
        e_c    = self._print(e)
        return '{} ** {}'.format(base_c, e_c)

    def _print_NumpySqrt(self, expr):
        arg = expr.args[0]
        dtype = arg.dtype.primitive_type
        if isinstance(dtype, (PrimitiveIntegerType, PrimitiveBooleanType)):
            arg = NumpyFloat(arg)
        code_args = self._print(arg)
        code = f'sqrt({code_args})'
        return code

    def _print_LiteralImaginaryUnit(self, expr):
        """ purpose: print complex numbers nicely in Fortran."""
        return "cmplx(0,1, kind = {})".format(self.print_kind(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_Literal(self, expr):
        printed = repr(expr.python_value)
        return "{}_{}".format(printed, self.print_kind(expr))

    def _print_LiteralTrue(self, expr):
        return ".True._{}".format(self.print_kind(expr))

    def _print_LiteralFalse(self, expr):
        return ".False._{}".format(self.print_kind(expr))

    def _print_LiteralComplex(self, expr):
        real_str = self._print(expr.real)
        imag_str = self._print(expr.imag)
        return "({}, {})".format(real_str, imag_str)

    def _print_IndexedElement(self, expr):
        base = expr.base

        inds = list(expr.indices)
        if len(inds) == 1 and isinstance(inds[0], LiteralEllipsis):
            inds = [Slice(None,None)]*expr.rank

        # Condense all indices on homogeneous objects into one IndexedElement for printing
        # This should be removed when support for lists is added
        if isinstance(base, IndexedElement):
            while isinstance(base, IndexedElement) and isinstance(base.class_type, HomogeneousContainerType):
                inds = list(base.indices) + inds
                base = base.base

        rank = base.rank
        if len(inds)<rank:
            inds += [Slice(None,None)]*(rank-base.class_type.container_rank)

        base_code = self._print(base)

        if base.order != 'F':
            inds = inds[::-1]
        allow_negative_indexes = base.allows_negative_indexes

        for i, ind in enumerate(inds):
            _shape = PyccelArrayShapeElement(base, i if expr.base.order != 'C' else len(inds) - i - 1)
            if isinstance(ind, Slice):
                inds[i] = self._new_slice_with_processed_arguments(ind, _shape, allow_negative_indexes)
            elif isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(_shape, ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i] = ind[0]
                if allow_negative_indexes and not isinstance(ind, LiteralInteger):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                            PyccelAdd(_shape, ind, simplify = True), ind)

        if isinstance(base.class_type, HomogeneousListType):
            if any(isinstance(i, Slice) for i in inds):
                raise NotImplementedError("Slice indexing not implemented for lists")
            inds = [PyccelAdd(i, LiteralInteger(1), simplify=True) for i in inds]
            inds_code = ", ".join(self._print(i) for i in inds)
            return f"{base_code}%of({inds_code})"
        elif isinstance(base.class_type, (NumpyNDArrayType, HomogeneousTupleType)):
            inds_code = ", ".join(self._print(i) for i in inds)
            return f"{base_code}({inds_code})"
        else:
            errors.report(f"Don't know how to index type {base.class_type}",
                    symbol=expr, severity='fatal')
            return ''

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
        start = _slice.start
        stop = _slice.stop
        step = _slice.step

        # negative start and end in slice
        if isinstance(start, PyccelUnarySub) and isinstance(start.args[0], LiteralInteger):
            start = PyccelMinus(array_size, start.args[0], simplify = True)
        elif start is not None and allow_negative_index and not isinstance(start,LiteralInteger):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                        PyccelAdd(array_size, start, simplify = True), start)

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0], simplify = True)
        elif stop is not None and allow_negative_index and not isinstance(stop, LiteralInteger):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                        PyccelAdd(array_size, stop, simplify = True), stop)

        # negative step in slice
        if isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            stop = PyccelAdd(stop, LiteralInteger(1), simplify = True) if stop is not None else LiteralInteger(0)
            start = start if start is not None else PyccelMinus(array_size, LiteralInteger(1), simplify = True)

        # variable step in slice
        elif step and allow_negative_index and not isinstance(step, LiteralInteger):
            if start is None :
                start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    LiteralInteger(0), PyccelMinus(array_size , LiteralInteger(1), simplify = True))

            if stop is None :
                stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    PyccelMinus(array_size, LiteralInteger(1), simplify = True), LiteralInteger(0))
            else :
                stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)),
                    stop, PyccelAdd(stop, LiteralInteger(1), simplify = True))

        elif stop is not None:
            stop = PyccelMinus(stop, LiteralInteger(1), simplify = True)

        return Slice(start, stop, step)

    def _print_Slice(self, expr):
        if expr.start is None or  isinstance(expr.start, Nil):
            start = ''
        else:
            start = self._print(expr.start)
        if (expr.stop is None) or isinstance(expr.stop, Nil):
            stop = ''
        else:
            stop = self._print(expr.stop)
        if expr.step is not None :
            return '{0}:{1}:{2}'.format(start, stop, self._print(expr.step))
        return '{0}:{1}'.format(start, stop)

#=======================================================================================

    def _print_FunctionCall(self, expr):
        func = expr.funcdef

        f_name = self._print(expr.func_name if not expr.interface else expr.interface_name)
        for k, m in _default_methods.items():
            f_name = f_name.replace(k, m)
        args   = expr.args
        func_results  = [r.var for r in func.results]
        parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, (Assign, AliasAssign)))
        is_function =  len(func_results) == 1 and func_results[0].rank == 0

        if func.arguments and func.arguments[0].bound_argument:
            class_variable = args[0].value
            args = args[1:]
            if isinstance(class_variable, FunctionCall):
                base = class_variable.funcdef.results[0].var
                if (not self._additional_code):
                    self._additional_code = ''
                var = self.scope.get_temporary_variable(base)

                self._additional_code = self._additional_code + self._print(Assign(var, class_variable)) + '\n'
                f_name = f'{self._print(var)} % {f_name}'
            else:
                f_name = f'{self._print(class_variable)} % {f_name}'

        if (not self._additional_code):
            self._additional_code = ''
        if parent_assign:
            lhs = parent_assign[0].lhs
            if len(func_results) == 1:
                lhs_vars = {func_results[0]:lhs}
            else:
                lhs_vars = dict(zip(func_results,lhs))
            assign_args = []
            for a in args:
                key = a.keyword
                arg = a.value
                if arg in lhs_vars.values():
                    var = arg.clone(self.scope.get_new_name())
                    self.scope.insert_variable(var)
                    self._additional_code += self._print(Assign(var,arg))
                    newarg = var
                else:
                    newarg = arg
                assign_args.append(FunctionCallArgument(newarg, key))
            args = assign_args
            results = list(lhs_vars.values())
            if is_function:
                results_strs = []
            else:
                # If func body is unknown then we may not know result names
                use_names = (len(func.body.body) != 0)
                if use_names:
                    results_strs = [f'{self._print(n)} = {self._print(r)}'
                            for n,r in lhs_vars.items()]
                else:
                    results_strs = [self._print(r) for r in lhs_vars.values()]

        elif not is_function and len(func_results)!=0:
            results = [r.clone(name = self.scope.get_new_name()) \
                        for r in func_results]
            for var in results:
                self.scope.insert_variable(var)

            results_strs = [f'{self._print(n)} = {self._print(r)}' \
                            for n,r in zip(func_results, results)]

        else:
            results_strs = []

        if func.is_inline:
            if len(func_results)>1:
                code = self._handle_inline_func_call(expr, assign_lhs = results)
            else:
                code = self._handle_inline_func_call(expr)
        else:
            args_strs = [self._print(a) for a in args if not isinstance(a.value, Nil)]
            args_code = ', '.join(args_strs+results_strs)
            code = f'{f_name}({args_code})'
            if not is_function:
                code = f'call {code}\n'

        if not parent_assign:
            if is_function or len(func_results) == 0:
                return code
            else:
                self._additional_code += code
                if len(func_results) == 1:
                    return self._print(results[0])
                else:
                    return self._print(tuple(results))
        elif is_function:
            result_code = self._print(results[0])
            if isinstance(parent_assign[0], AliasAssign):
                return f'{result_code} => {code}\n'
            else:
                return f'{result_code} = {code}\n'
        else:
            return code

#=======================================================================================

    def _print_PyccelInternalFunction(self, expr):
        if isinstance(expr, NumpyNewArray):
            return errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')
        else:
            return self._print_not_supported(expr)

#=======================================================================================

    def _print_PrecomputedCode(self, expr):
        return expr.code

#=======================================================================================

    def _print_CLocFunc(self, expr):
        lhs = self._print(expr.result)
        rhs = self._print(expr.arg)
        self._constantImports.setdefault('ISO_C_Binding', set()).add('c_loc')
        return f'{lhs} = c_loc({rhs})\n'

#=======================================================================================

    def _print_C_F_Pointer(self, expr):
        self._constantImports.setdefault('ISO_C_Binding', set()).add('C_F_Pointer')
        shape = ','.join(self._print(s) for s in expr.shape)
        if shape:
            return f'call C_F_Pointer({self._print(expr.c_pointer)}, {self._print(expr.f_array)}, [{shape}])\n'
        else:
            return f'call C_F_Pointer({self._print(expr.c_pointer)}, {self._print(expr.f_array)})\n'

#=======================================================================================

    def _print_PythonConjugate(self, expr):
        return 'conjg( {} )'.format( self._print(expr.internal_var) )

#=======================================================================================

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (ending with a \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos

        # split line by line and add the splitted lines to result
        result = []
        trailing = ' &'
        # trailing with no added space characters in case splitting is within quotes
        quote_trailing = '&'

        for line in lines:
            if len(line) > 72:
                cline = line[:72].lstrip()
                if cline.startswith('!') and not cline.startswith('!$'):
                    result.append(line)
                    continue

                tab_len = line.index(cline[0])
                # code line
                # set containing positions inside quotes
                inside_quotes_positions = set()
                inside_quotes_intervals = [(match.start(), match.end())
                                           for match in re.compile('("[^"]*")|(\'[^\']*\')').finditer(line)]
                for lidx, ridx in inside_quotes_intervals:
                    for idx in range(lidx, ridx):
                        inside_quotes_positions.add(idx)
                initial_len = len(line)
                pos = split_pos_code(line, 72)

                startswith_omp = cline.startswith('!$omp')
                startswith_acc = cline.startswith('!$acc')

                if startswith_acc or startswith_omp:
                    assert pos>=5

                if pos not in inside_quotes_positions:
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                else:
                    hunk = line[:pos]
                    line = line[pos:]

                if line:
                    hunk += (quote_trailing if pos in inside_quotes_positions else trailing)

                last_cut_was_inside_quotes = pos in inside_quotes_positions
                result.append(hunk)
                while len(line) > 0:
                    removed = initial_len - len(line)
                    pos = split_pos_code(line, 65-tab_len)
                    if pos + removed not in inside_quotes_positions:
                        hunk = line[:pos].rstrip()
                        line = line[pos:].lstrip()
                    else:
                        hunk = line[:pos]
                        line = line[pos:]
                    if line:
                        hunk += (quote_trailing if (pos + removed) in inside_quotes_positions else trailing)

                    if last_cut_was_inside_quotes:
                        hunk_start = tab_len*' ' + '&'
                    elif startswith_omp:
                        hunk_start = tab_len*' ' + '!$omp &'
                    elif startswith_acc:
                        hunk_start = tab_len*' ' + '!$acc &'
                    else:
                        hunk_start = tab_len*' ' + '      '

                    result.append(hunk_start + hunk)
                    last_cut_was_inside_quotes = (pos + removed) in inside_quotes_positions
            else:
                result.append(line)

        # make sure that all lines end with a carriage return
        return [l if l.endswith('\n') else l+'\n' for l in result]

    def indent_code(self, code):
        """
        Add the correct indentation to the code.

        Analyse the code to calculate when indentation is needed.
        Add the necessary spaces at the start of each line.

        Parameters
        ----------
        code : str | iterable[str]
            A string of code or a list of code lines.

        Returns
        -------
        list[str]
            A list of indented code lines.
        """
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [line.lstrip(' \t') for line in code]

        increase = [int(inc_regex.match(line) is not None)
                     for line in code]
        decrease = [int(dec_regex.match(line) is not None)
                     for line in code]

        level = 0
        tabwidth = self._default_settings['tabwidth']
        new_code = []
        for i, line in enumerate(code):
            if line in ('','\n') or line.startswith('#'):
                new_code.append(line)
                continue
            level -= decrease[i]

            padding = " "*(level*tabwidth)

            line = "%s%s" % (padding, line)

            new_code.append(line)
            level += increase[i]

        return new_code

    def _print_BindCArrayVariable(self, expr):
        return self._print(expr.wrapper_function)

    def _print_BindCClassDef(self, expr):
        funcs = [expr.new_func, *expr.methods, *[f for i in expr.interfaces for f in i.functions],
                 *[a.getter for a in expr.attributes], *[a.setter for a in expr.attributes]]
        sep = f'\n{self._print(SeparatorComment(40))}\n'
        return '', sep.join(self._print(f) for f in funcs)

    def _print_MacroDefinition(self, expr):
        name = expr.macro_name
        obj = self._print(expr.object)
        return f'#define {name} {obj}\n'

    def _print_KindSpecification(self, expr):
        return f'(kind = {self.print_kind(expr.type_specifier)})'
