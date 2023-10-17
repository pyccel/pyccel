# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""


import string
import re
from itertools import chain
from collections import OrderedDict

import functools

from pyccel.ast.basic import TypedAstNode

from pyccel.ast.bind_c import BindCPointer, BindCFunctionDef, BindCFunctionDefArgument, BindCModule

from pyccel.ast.builtins import PythonInt, PythonType,PythonPrint, PythonRange
from pyccel.ast.builtins import PythonFloat, PythonTuple
from pyccel.ast.builtins import PythonComplex, PythonBool, PythonAbs
from pyccel.ast.builtins import python_builtin_datatypes_dict as python_builtin_datatypes

from pyccel.ast.core import get_iterable_ranges
from pyccel.ast.core import FunctionDef, InlineFunctionDef
from pyccel.ast.core import SeparatorComment, Comment
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import FunctionCallArgument
from pyccel.ast.core import ErrorExit, FunctionAddress
from pyccel.ast.core import Return, Module, If, IfSection, For
from pyccel.ast.core import Import, CodeBlock, AsName, EmptyNode
from pyccel.ast.core import Assign, AliasAssign, Declare, Deallocate
from pyccel.ast.core import FunctionCall, DottedFunctionCall, PyccelFunctionDef

from pyccel.ast.datatypes import is_pyccel_datatype
from pyccel.ast.datatypes import is_iterable_datatype, is_with_construct_datatype
from pyccel.ast.datatypes import NativeSymbol, NativeString, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeFloat, NativeComplex
from pyccel.ast.datatypes import iso_c_binding
from pyccel.ast.datatypes import iso_c_binding_shortcut_mapping
from pyccel.ast.datatypes import NativeRange, NativeNumeric
from pyccel.ast.datatypes import CustomDataType

from pyccel.ast.internals import Slice, PrecomputedCode, PyccelArrayShapeElement
from pyccel.ast.internals import PyccelInternalFunction, get_final_precision

from pyccel.ast.itertoolsext import Product

from pyccel.ast.literals  import LiteralInteger, LiteralFloat, Literal
from pyccel.ast.literals  import LiteralTrue, LiteralFalse, LiteralString
from pyccel.ast.literals  import Nil

from pyccel.ast.mathext  import math_constants

from pyccel.ast.numpyext import NumpyEmpty, NumpyInt32
from pyccel.ast.numpyext import NumpyFloat, NumpyBool
from pyccel.ast.numpyext import NumpyReal, NumpyImag
from pyccel.ast.numpyext import NumpyRand
from pyccel.ast.numpyext import NumpyNewArray
from pyccel.ast.numpyext import NumpyNonZero
from pyccel.ast.numpyext import NumpySign
from pyccel.ast.numpyext import DtypePrecisionToCastFunction

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus
from pyccel.ast.operators import PyccelMod
from pyccel.ast.operators import PyccelUnarySub, PyccelLt, PyccelGt, IfTernaryOperator

from pyccel.ast.utilities import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities import expand_to_loops

from pyccel.ast.variable import Variable, IndexedElement, InhomogeneousTupleVariable, DottedName

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *
from pyccel.codegen.printing.codeprinter import CodePrinter


# TODO: add examples
# TODO: use _get_statement when returning a string

__all__ = ["FCodePrinter", "fcode"]

numpy_ufunc_to_fortran = {
    'NumpyAbs'  : 'abs',
    'NumpyFabs'  : 'abs',
    'NumpyMin'  : 'minval',
    'NumpyAmin'  : 'minval',
    'NumpyAmax'  : 'maxval',
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
    '__init__': 'create',
    '__del__' : 'free',
}

type_to_print_format = {
        ('float'): 'F0.12',
        ('complex'): '"(",F0.12," + ",F0.12,")"',
        ('int'): 'I0',
        ('bool'): 'A',
        ('string'): 'A',
        ('tuple'):  '*'
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

        errors.set_target(filename, 'file')

        super().__init__()
        self._constantImports = {}
        self._current_class    = None

        self._additional_code = None
        self._additional_imports = set()

        self.prefix_module = prefix_module

    def print_constant_imports(self):
        """Prints the use line for the constant imports used"""
        macros = []
        for (name, imports) in self._constantImports.items():

            macro = f"use, intrinsic :: {name}, only : "
            rename = [c if isinstance(c, str) else c[0] + ' => ' + c[1] for c in imports]
            if len(rename) == 0:
                continue
            macro += " , ".join(rename)
            macros.append(macro)
        return "\n".join(macros)

    def get_additional_imports(self):
        """return the additional modules collected for importing in printing stage"""
        return [i.source for i in self._additional_imports]

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

    def _get_statement(self, codestring):
        return codestring

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def print_kind(self, expr):
        """
        Prints the kind(precision) of a literal value or its shortcut if possible
        """
        precision = get_final_precision(expr)
        constant_name = iso_c_binding[self._print(expr.dtype)][precision]
        constant_shortcut = iso_c_binding_shortcut_mapping[constant_name]
        if constant_shortcut not in self.scope.all_used_symbols and constant_name != constant_shortcut:
            self._constantImports.setdefault('ISO_C_Binding', set())\
                .add((constant_shortcut, constant_name))
            constant_name = constant_shortcut
        else:
            self._constantImports.setdefault('ISO_C_Binding', set())\
                .add(constant_name)
        return constant_name

    def _handle_inline_func_call(self, expr, provided_args, assign_lhs = None):
        """ Print a function call to an inline function
        """
        scope = self.scope
        func = expr.funcdef

        # Print any arguments using the same inline function
        # As the function definition is modified directly this function
        # cannot be called recursively with the same FunctionDef
        args = []
        for a in provided_args:
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

        self._additional_imports.update(func.imports)
        if func.global_vars or func.global_funcs:
            mod = func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            self._additional_imports.add(Import(mod.name, [AsName(v, v.name) \
                for v in (*func.global_vars, *func.global_funcs)]))
            for v in (*func.global_vars, *func.global_funcs):
                self.scope.insert_symbol(v.name)

        self.set_scope(scope)
        return code

    def _get_external_declarations(self):
        """
        Find external functions and declare their result type.

        Look for any external functions in the local imports from
        the scope and use their definitions to create declarations
        from the results. These declarations are stored in a
        dictionary whose keys are the result variable which will
        be declared.

        Returns
        -------
        dict
            The declarations necessary to use the external function.
        """
        decs = {}
        for key,f in self.scope.imports['functions'].items():
            if isinstance(f, FunctionDef) and f.is_external:
                i = Variable(f.results[0].var.dtype, name=str(key))
                dec = Declare(i.dtype, i, external=True)
                decs[i] = dec

        return decs

    # ============ Elements ============ #
    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        if isinstance(expr.name, AsName):
            name = self._print(expr.name.target)
        else:
            name = self._print(expr.name)
        name = name.replace('.', '_')
        if not name.startswith('mod_') and self.prefix_module:
            name = '{prefix}_{name}'.format(prefix=self.prefix_module,
                                            name=name)

        imports = ''.join(self._print(i) for i in expr.imports)

        # Define declarations
        decs = ''
        # ...
        class_decs_and_methods = [self._print(i) for i in expr.classes]
        decs += '\n'.join(c[0] for c in class_decs_and_methods)
        # ...

        decs += ''.join(self._print(i) for i in expr.declarations)
        # look for external functions and declare their result type
        external_decs = self._get_external_declarations()
        decs += ''.join(self._print(i) for i in external_decs.values())

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

        # Get class functions
        func_strings = [c[1] for c in class_decs_and_methods]
        if expr.funcs:
            func_strings += [''.join([sep, self._print(i), sep]) for i in expr.funcs]
        if isinstance(expr, BindCModule):
            func_strings += [''.join([sep, self._print(i), sep]) for i in expr.variable_wrappers]
        body = '\n'.join(func_strings)
        # ...

        contains = 'contains\n' if (expr.funcs or expr.classes or expr.interfaces) else ''
        imports += ''.join(self._print(i) for i in self._additional_imports)
        imports += "\n" + self.print_constant_imports()
        parts = ['module {}\n'.format(name),
                 imports,
                 'implicit none\n',
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
        decs = ''.join(self._print_Declare(Declare(v.dtype, v)) for v in variables)

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
        imports += ''.join(self._print(i) for i in self._additional_imports)
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

        if isinstance(expr.source, AsName):
            source = expr.source.target
        else:
            source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1]
        else:
            source = self._print(source)

        # importing of pyccel extensions is not printed
        if source in pyccel_builtin_import_registry:
            return ''

        if expr.source_module:
            source = expr.source_module.scope.get_expected_name(source)

        if 'mpi4py' == str(getattr(expr.source,'name',expr.source)):
            return 'use mpi\n' + 'use mpiext\n'

        targets = [t for t in expr.target if not isinstance(t.object, Module)]

        if len(targets) == 0:
            return 'use {}\n'.format(source)

        targets = [t for t in targets if not isinstance(t.object, InlineFunctionDef)]
        if len(targets) == 0:
            return ''

        prefix = 'use {}, only:'.format(source)

        code = ''
        for i in targets:
            old_name = i.name
            new_name = i.target
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
        return self._get_statement(code) + '\n'

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
            if isinstance(f, (InhomogeneousTupleVariable, PythonTuple, str)):
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
            elif isinstance(f.rank, int) and f.rank > 0:
                if args_format:
                    code += self._formatted_args_to_print(args_format, args, sep, separator, expr)
                    args_format = []
                    args = []
                loop_scope = self.scope.create_new_loop_scope()
                for_index = self.scope.get_temporary_variable(NativeInteger(), name='i')
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

    def _get_print_format_and_arg(self,var):
        """ Get the format string and the printable argument for an object.
        In other words get arg_format and arg such that var can be printed
        by doing:

        > write(*, arg_format) arg

        Parameters
        ----------
        var : TypedAstNode
              The object to be printed

        Results
        -------
        arg_format : str
                     The format string
        arg        : str
                     The fortran code which represents var
        """
        var_type = var.dtype
        try:
            arg_format = type_to_print_format[str(var_type)]
        except KeyError:
            errors.report("{} type is not supported currently".format(var_type), severity='fatal')

        if var_type is NativeComplex():
            arg = '{}, {}'.format(self._print(NumpyReal(var)), self._print(NumpyImag(var)))
        elif var_type is NativeBool():
            if isinstance(var, LiteralTrue):
                arg = "'True'"
            elif isinstance(var, LiteralFalse):
                arg = "'False'"
            else:
                arg = 'merge("True ", "False", {})'.format(self._print(var))
        else:
            arg = self._print(var)

        return arg_format, arg

    def _print_SymbolicPrint(self, expr):
        # for every expression we will generate a print
        code = '\n'.join("print *, 'sympy> {}'".format(a) for a in expr.expr)
        return self._get_statement(code) + '\n'

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
        """ print the python builtin function abs
        args : variable
        """
        return "abs({})".format(self._print(expr.arg))

    def _print_PythonTuple(self, expr):
        shape = tuple(reversed(expr.shape))
        if len(shape)>1:
            elements = ', '.join(self._print(i) for i in expr)
            shape    = ', '.join(self._print(i) for i in shape)
            return 'reshape(['+ elements + '], '+ '[' + shape + ']' + ')'
        fs = ', '.join(self._print(f) for f in expr)
        return '[{0}]'.format(fs)

    def _print_PythonList(self, expr):
        return self._print_PythonTuple(expr)

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

    def _print_PythonLen(self, expr):
        var = expr.arg
        idx = 1 if var.order == 'F' else var.rank
        prec = self.print_kind(expr)

        dtype = var.dtype
        if dtype is NativeString():
            return 'len({})'.format(self._print(var))
        elif var.rank == 1:
            return 'size({}, kind={})'.format(self._print(var), prec)
        else:
            return 'size({},{},{})'.format(self._print(var), self._print(idx), prec)

    def _print_PythonSum(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return "sum({})".format(", ".join(args))

    def _print_PythonReal(self, expr):
        value = self._print(expr.internal_var)
        return 'real({0})'.format(value)

    def _print_PythonImag(self, expr):
        value = self._print(expr.internal_var)
        return 'aimag({0})'.format(value)



    #========================== Numpy Elements ===============================#

    def _print_NumpySum(self, expr):
        """Fortran print."""

        rhs_code = self._print(expr.arg)
        if isinstance(expr.arg.dtype, NativeBool):
            return 'count({0})'.format(rhs_code)
        return 'sum({0})'.format(rhs_code)

    def _print_NumpyProduct(self, expr):
        """Fortran print."""

        rhs_code = self._print(expr.arg)
        return 'product({0})'.format(rhs_code)

    def _print_NumpyMatmul(self, expr):
        """Fortran print."""
        a_code = self._print(expr.a)
        b_code = self._print(expr.b)

        if expr.rank == 0:
            if isinstance(expr.a.dtype, NativeBool):
                a_code = self._print(PythonInt(expr.a))
            if isinstance(expr.b.dtype, NativeBool):
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
        """Fortran print."""
        arg = PythonAbs(expr.arg) if isinstance(expr.arg.dtype, NativeComplex) else expr.arg
        if expr.axis:
            axis = expr.axis
            if arg.order != 'F':
                axis = PyccelMinus(LiteralInteger(arg.rank), expr.axis, simplify=True)
            else:
                axis = LiteralInteger(expr.axis.python_value + 1)
            code = 'Norm2({},{})'.format(self._print(arg), self._print(axis))
        else:
            code = 'Norm2({})'.format(self._print(arg))

        return code

    def _print_NumpyLinspace(self, expr):

        if expr.stop.dtype != expr.dtype or expr.precision != expr.stop.precision:
            cast_func = DtypePrecisionToCastFunction[expr.dtype.name][expr.precision]
            st = cast_func(expr.stop)
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
            var = self.scope.get_temporary_variable('int', 'linspace_index') 

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

        ind   = self._print(self.scope.get_temporary_variable('int'))
        array = expr.array

        if array.dtype is NativeBool():
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

        if array.dtype is NativeBool():
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
        value_true  = expr.value_true
        value_false = expr.value_false
        try :
            cast_func = DtypePrecisionToCastFunction[expr.dtype.name][expr.precision]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        if value_true.dtype != expr.dtype or value_true.precision != expr.precision:
            value_true = cast_func(value_true)
        if value_false.dtype != expr.dtype or value_false.precision != expr.precision:
            value_false = cast_func(value_false)

        condition   = self._print(expr.condition)
        value_true  = self._print(value_true)
        value_false = self._print(value_false)

        stmt = 'merge({true}, {false}, {cond})'.format(
                true=value_true,
                false=value_false,
                cond=condition)

        return stmt

    def _print_NumpyArray(self, expr):
        expr_args = (expr.arg,) if isinstance(expr.arg, Variable) else expr.arg
        order = expr.order
        # If Numpy array is stored with column-major ordering, transpose values
        # use reshape with order for rank > 2
        if expr.rank <= 2:
            rhs_code = self._print(expr.arg)
            if expr.arg.order and expr.arg.order != expr.order:
                rhs_code = f'transpose({rhs_code})'
            if expr.arg.rank < expr.rank:
                if order == 'F':
                    shape_code = ', '.join(self._print(i) for i in expr.shape)
                else:
                    shape_code = ', '.join(self._print(i) for i in expr.shape[::-1])
                rhs_code = f"reshape({rhs_code}, [{shape_code}])"
        else:
            new_args = []
            inv_order = 'C' if order == 'F' else 'F'
            for a in expr_args:
                ac = self._print(a)
                if a.order == inv_order:
                    shape = a.shape if a.order == 'C' else a.shape[::-1]
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
        index  = self.scope.get_temporary_variable('int')

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
        init_value = self._print(expr.arg)
        prec = self.print_kind(expr)

        if expr.arg.rank == 1:
            return f'size({init_value}, kind={prec})'

        if expr.arg.order == 'C':
            index = PyccelMinus(LiteralInteger(expr.arg.rank), expr.index, simplify = True)
            index = self._print(index)
        else:
            index = PyccelAdd(expr.index, LiteralInteger(1), simplify = True)
            index = self._print(index)

        return f'size({init_value}, {index}, {prec})'

    def _print_PythonInt(self, expr):
        value = self._print(expr.arg)
        if (expr.arg.dtype is NativeBool()):
            code = 'MERGE(1_{0}, 0_{1}, {2})'.format(self.print_kind(expr), self.print_kind(expr),value)
        else:
            code  = 'Int({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        if (expr.arg.dtype is NativeBool()):
            code = 'MERGE(1.0_{0}, 0.0_{1}, {2})'.format(self.print_kind(expr), self.print_kind(expr),value)
        else:
            code  = 'Real({0}, {1})'.format(value, self.print_kind(expr))
        return code

    def _print_MathFloor(self, expr):
        arg = expr.args[0]
        arg_code = self._print(arg)

        # math.floor on integer argument is identity,
        # but we need parentheses around expressions
        if arg.dtype is NativeInteger():
            return '({})'.format(arg_code)

        prec = expr.precision
        prec_code = self.print_kind(expr)
        return 'floor({}, kind={})'.format(arg_code, prec_code)

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            var = self._print(expr.internal_var)
            code = 'cmplx({0}, kind={1})'.format(var,
                                self.print_kind(expr))
        else:
            real = self._print(expr.real)
            imag = self._print(expr.imag)
            code = 'cmplx({0}, {1}, {2})'.format(real, imag,
                                self.print_kind(expr))
        return code

    def _print_PythonBool(self, expr):
        if isinstance(expr.arg.dtype, NativeBool):
            return 'logical({}, kind = {prec})'.format(self._print(expr.arg), prec = self.print_kind(expr))
        else:
            return '({} /= 0)'.format(self._print(expr.arg))

    def _print_NumpyRand(self, expr):
        if expr.rank != 0:
            errors.report(FORTRAN_ALLOCATABLE_IN_EXPRESSION,
                          symbol=expr, severity='fatal')

        if (not self._additional_code):
            self._additional_code = ''
        var = self.scope.get_temporary_variable(expr.dtype, memory_handling = 'stack',
                shape = expr.shape, precision = expr.precision,
                order = expr.order, rank = expr.rank)

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

    def _print_PythonMin(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'minval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'min('+code+')'
        return self._get_statement(code)

    def _print_PythonMax(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            code = 'maxval({0})'.format(self._print(arg))
        else:
            code = ','.join(self._print(arg) for arg in args)
            code = 'max('+code+')'
        return self._get_statement(code)

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
        prec  = expr.argument.precision

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
        if isinstance(expr.dtype, NativeSymbol):
            return ''

        if isinstance(expr.dtype, NativeRange):
            return ''

        # meta-variables
        if (isinstance(expr.variable, Variable) and
            expr.variable.name.startswith('__')):
            return ''
        # ...

        if isinstance(expr.variable, InhomogeneousTupleVariable):
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        # ... TODO improve
        # Group the variables by intent
        var = expr.variable
        expr_dtype = expr.dtype
        rank            = var.rank
        shape           = var.alloc_shape
        is_const        = var.is_const
        is_optional     = var.is_optional
        is_private      = var.is_private
        is_alias        = var.is_alias and not isinstance(expr_dtype, BindCPointer)
        on_heap         = var.on_heap
        on_stack        = var.on_stack
        is_static       = expr.static
        is_external     = expr.external
        is_target       = var.is_target and not var.is_alias
        intent          = expr.intent
        intent_in = intent and intent != 'out'

        if isinstance(shape, (tuple,PythonTuple)) and len(shape) ==1:
            shape = shape[0]
        # ...

        # ... print datatype
        if isinstance(expr.dtype, CustomDataType):
            name   = expr_dtype.__class__.__name__
            prefix = expr_dtype.prefix
            alias  = expr_dtype.alias

            if var.is_argument:
                sig = 'class'
            else:
                sig = 'type'

            if alias is None:
                name = name.replace(prefix, '')
            else:
                name = alias
            dtype = f'{sig}({name})'
        else:
            dtype = self._print(expr_dtype)

        # ...
            if isinstance(expr_dtype, NativeString):

                if intent_in:
                    dtype = dtype[:9] +'(len =*)'
                    #TODO improve ,this is the case of character as argument
            elif isinstance(expr_dtype, BindCPointer):
                dtype = 'type(c_ptr)'
                self._constantImports.setdefault('ISO_C_Binding', set()).add('c_ptr')
            else:
                dtype += '({0})'.format(self.print_kind(expr.variable))

        code_value = ''
        if expr.value:
            code_value = ' = {0}'.format(self._print(expr.value))

        vstr = self._print(expr.variable.name)

        # arrays are 0-based in pyccel, to avoid ambiguity with range
        s = self._print(LiteralInteger(0))
        if not(is_static) and (on_heap or (var.shape is None)):
            s = ''

        # Default empty strings
        intentstr      = ''
        allocatablestr = ''
        optionalstr    = ''
        privatestr     = ''
        rankstr        = ''
        externalstr    = ''

        # Compute intent string
        if intent:
            if intent == 'in' and rank == 0 and not (is_static and is_optional):
                intentstr = ', value'
                if is_const:
                    intentstr += ', intent(in)'
            else:
                intentstr = ', intent({})'.format(intent)

        # Compute allocatable string
        if not is_static:
            if is_alias:
                allocatablestr = ', pointer'

            elif on_heap and not intent_in:
                allocatablestr = ', allocatable'

            # ISSUES #177: var is allocatable and target
            if is_target:
                allocatablestr = '{}, target'.format(allocatablestr)

        # Compute optional string
        if is_optional:
            optionalstr = ', optional'

        # Compute private string
        if is_private:
            privatestr = ', private'

        # Compute external string
        if is_external:
            externalstr = ', external'

        # Compute rank string
        # TODO: improve
        if ((rank == 1) and (isinstance(shape, (int, TypedAstNode))) and (is_static or on_stack)):
            rankstr = '({0}:{1})'.format(self._print(s), self._print(PyccelMinus(shape, LiteralInteger(1), simplify = True)))

        elif ((rank > 0) and (isinstance(shape, (PythonTuple, tuple))) and (is_static or on_stack)):
            #TODO fix bug when we include shape of type list

            if var.order == 'C':
                rankstr = ','.join('{0}:{1}'.format(self._print(s),
                                                      self._print(PyccelMinus(i, LiteralInteger(1), simplify = True))) for i in shape[::-1])
            else:
                rankstr =  ','.join('{0}:{1}'.format(self._print(s),
                                                     self._print(PyccelMinus(i, LiteralInteger(1), simplify = True))) for i in shape)
            rankstr = '({rank})'.format(rank=rankstr)

        elif (rank > 0) and on_heap and intent_in:
            rankstr = '({})'.format(','.join(['0:'] * rank))

        elif (rank > 0) and (on_heap or is_alias):
            rankstr = '({})'.format(','.join( [':'] * rank))

#        else:
#            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
#                severity='fatal')

        mod_str = ''
        if expr.module_variable and not is_private and (rank == 0) and expr_dtype in NativeNumeric:
            mod_str = ', bind(c)'

        # Construct declaration
        left  = dtype + allocatablestr + optionalstr + privatestr + externalstr + mod_str + intentstr
        right = vstr + rankstr + code_value
        return '{} :: {}\n'.format(left, right)

    def _print_AliasAssign(self, expr):
        code = ''
        lhs = expr.lhs
        rhs = expr.rhs

        if isinstance(lhs, InhomogeneousTupleVariable):
            return self._print(CodeBlock([AliasAssign(l, r) for l,r in zip(lhs,rhs)]))

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

        return self._get_statement(code) + '\n'

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
              expr.lhs.dtype == NativeSymbol()):
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
        return self._get_statement(code) + '\n'

#------------------------------------------------------------------------------
    def _print_Allocate(self, expr):

        # Transpose indices because of Fortran column-major ordering
        shape = expr.shape if expr.order == 'F' else expr.shape[::-1]

        var_code = self._print(expr.variable)
        size_code = ', '.join(self._print(i) for i in shape)
        shape_code = ', '.join('0:' + self._print(PyccelMinus(i, LiteralInteger(1), simplify = True)) for i in shape)
        code = ''

        if expr.status == 'unallocated':
            code += 'allocate({0}({1}))\n'.format(var_code, shape_code)

        elif expr.status == 'unknown':
            code += 'if (allocated({})) then\n'.format(var_code)
            code += '  if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '    deallocate({})\n'     .format(var_code)
            code += '    allocate({0}({1}))\n'.format(var_code, shape_code)
            code += '  end if\n'
            code += 'else\n'
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        elif expr.status == 'allocated':
            code += 'if (any(size({}) /= [{}])) then\n'.format(var_code, size_code)
            code += '  deallocate({})\n'     .format(var_code)
            code += '  allocate({0}({1}))\n'.format(var_code, shape_code)
            code += 'end if\n'

        return code

#-----------------------------------------------------------------------------
    def _print_Deallocate(self, expr):
        var = expr.variable
        if isinstance(var, InhomogeneousTupleVariable):
            return ''.join(self._print(Deallocate(v)) for v in var)

        if isinstance(var.dtype, CustomDataType):
            Pyccel__del = expr.variable.cls_base.scope.find('__del__')
            Pyccel_del_args = [FunctionCallArgument(arg) for arg in Pyccel__del.arguments]
            return self._print(DottedFunctionCall(Pyccel__del, Pyccel_del_args, var))

        if var.is_alias:
            return ''
        else:
            var_code = self._print(var)
            code  = 'if (allocated({})) then\n'.format(var_code)
            code += '  deallocate({})\n'     .format(var_code)
            code += 'end if\n'
            return code
#------------------------------------------------------------------------------

    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeFloat(self, expr):
        return 'real'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_NativeString(self, expr):
        return 'character(len=280)'
        #TODO fix improve later

    def _print_DataType(self, expr):
        return self._print(expr.name)

    def _print_LiteralString(self, expr):
        sp_chars = ['\a', '\b', '\f', '\r', '\t', '\v', "'", '\n']
        sub_str = ''
        formatted_str = []
        for c in expr.arg:
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
        # ... we don't print 'hidden' functions
        name = self._print(expr.name)
        if all(isinstance(f, FunctionAddress) for f in expr.functions):
            funcs = expr.functions
        else:
            funcs = [f for f in expr.functions if f is \
                    expr.point([FunctionCallArgument(a.var.clone('arg_'+str(i))) \
                        for i,a in enumerate(f.arguments)], use_final_precision = True)]

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

    def _print_Block(self, expr):
        self.set_scope(expr.scope)

        decs=[]
        for i in expr.variables:
            dec = Declare(i.dtype, i)
            decs += [dec]
        body = expr.body

        body_code = self._print(body)
        prelude   = ''.join(self._print(i) for i in decs)

        self.exit_scope()

        #case of no local variables
        if len(decs) == 0:
            return body_code

        return ('{name} : Block\n'
                '{prelude}\n'
                 '{body}\n'
                'end Block {name}\n').format(name=expr.name, prelude=prelude, body=body_code)

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
                args_decs[result] = Declare(result.dtype, result, intent='out')

            functions = expr.functions

        else:
           #todo: if return is a function
            func_type = 'function'
            result = out_args[0]
            functions = expr.functions

            func_end = 'result({0})'.format(result.name)

            args_decs[result] = Declare(result.dtype, result)
            out_args = []
        # ...

        for i, arg in enumerate(arguments):
            arg_var = arg.var
            if isinstance(arg_var, Variable):
                if isinstance(arg, BindCFunctionDefArgument) and arg.original_function_argument_variable.rank!=0:
                    for b_arg,inout in zip(arg.get_all_function_def_arguments(), arg.inout):
                        v = b_arg.var
                        if inout:
                            dec = Declare(v.dtype, v, intent='inout')
                        else:
                            dec = Declare(v.dtype, v, intent='in')
                        args_decs[v] = dec
                else:
                    if i == 0 and expr.cls_name:
                        dec = Declare(arg_var.dtype, arg_var, intent='inout', passed_from_dotted = True)
                    elif arg.inout:
                        dec = Declare(arg_var.dtype, arg_var, intent='inout')
                    else:
                        dec = Declare(arg_var.dtype, arg_var, intent='in')
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


        name = self._print(expr.name)

        if expr.cls_name:
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

        sig_parts = self.function_signature(expr, name)
        bind_c = ' bind(c)' if isinstance(expr, BindCFunctionDef) else ''
        prelude = sig_parts.pop('arg_decs')
        decs = OrderedDict()
        functions = [f for f in expr.functions if not f.is_inline]
        func_interfaces = '\n'.join(self._print(i) for i in expr.interfaces)
        body_code = self._print(expr.body)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        for i in expr.local_vars:
            dec = Declare(i.dtype, i)
            decs[i] = dec

        decs.update(self._get_external_declarations())

        arguments = [a.var for a in expr.arguments]
        results = [a.var for a in expr.results]
        vars_to_print = self.scope.variables.values()
        for v in vars_to_print:
            if (v not in expr.local_vars) and (v not in results) and (v not in arguments):
                decs[v] = Declare(v.dtype,v)
        prelude += ''.join(self._print(i) for i in decs.values())
        if len(functions)>0:
            functions_code = '\n'.join(self._print(i) for  i in functions)
            body_code = body_code +'\ncontains\n' + functions_code

        imports = ''.join(self._print(i) for i in expr.imports)

        parts = [doc_string,
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

        decs = ''.join(self._print(Declare(i.dtype, i)) for i in expr.attributes)

        aliases = []
        names   = []
        ls = [self._print(i.name) for i in expr.methods]
        for i in ls:
            j = _default_methods.get(i,i)
            aliases.append(j)
            names.append('{0}_{1}'.format(name, self._print(j)))
        methods = ''.join('procedure :: {0} => {1}\n'.format(i, j) for i, j in zip(aliases, names))
        for i in expr.interfaces:
            names = ','.join('{0}_{1}'.format(name, self._print(j.name)) for j in i.functions)
            methods += 'generic, public :: {0} => {1}\n'.format(self._print(i.name), names)
            methods += 'procedure :: {0}\n'.format(names)



        options = ', '.join(i for i in expr.options)

        sig = 'type, {0}'.format(options)
        if not(base is None):
            sig = '{0}, extends({1})'.format(sig, base)

        code = ('{0} :: {1}').format(sig, name)
        if len(decs) > 0:
            code = ('{0}\n'
                    '{1}').format(code, decs)
        if len(methods) > 0:
            code = ('{0}\n'
                    'contains\n'
                    '{1}').format(code, methods)
        decs = ('{0}\n'
                'end type {1}\n').format(code, name)

        sep = self._print(SeparatorComment(40))
        # we rename all methods because of the aliasing
        cls_methods = [i.clone('{0}'.format(i.name)) for i in expr.methods]
        for i in expr.interfaces:
            cls_methods +=  [j.clone('{0}'.format(j.name)) for j in i.functions]

        methods = ''
        for i in cls_methods:
            methods = ('{methods}\n'
                     '{sep}\n'
                     '{f}\n'
                     '{sep}\n').format(methods=methods, sep=sep, f=self._print(i))

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
        step  = self._print(expr.step)

        test_step = expr.step
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
        return '{0}, {1}, {2}'.format(start, stop, step)

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
        prolog = '!$acc parallel {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_ACC_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        # ...

        # ... TODO adapt get_statement to have continuation with OpenACC
        prolog = '!$acc loop {clauses}\n'.format(clauses=clauses)
        epilog = '!$acc end loop\n'
        # ...

        # ...
        code = ('{prolog}'
                '{loop}'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

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

    def _print_ForIterator(self, expr):
        return self._print_For(expr)

        prolog = ''
        epilog = ''

        # ...
        def _do_range(target, iterable, prolog, epilog):
            tar        = self._print(target)
            range_code = self._print(iterable)

            prolog += 'do {0} = {1}\n'.format(tar, range_code)
            epilog = 'end do\n' + epilog

            return prolog, epilog
        # ...

        # ...
        if not isinstance(expr.iterable, (Variable, ConstructorCall)):
            raise TypeError('iterable must be Variable or ConstructorCall.')
        # ...

        # ...
        targets = expr.target
        if isinstance(expr.iterable, Variable):
            iters = expr.ranges
        elif isinstance(expr.iterable, ConstructorCall):
            iters = get_iterable_ranges(expr.iterable)
        # ...

        # ...
        for i,a in zip(targets, iters):
            prolog, epilog = _do_range(i, a, \
                                       prolog, epilog)

        body = ''.join(self._print(i) for i in expr.body)
        # ...

        return ('{prolog}'
                '{body}'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)


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

        if (a.dtype is NativeBool() and b.dtype is NativeBool()):
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

        if a.dtype is NativeBool() and b.dtype is NativeBool():
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

        cond = PythonBool(expr.cond) if not isinstance(expr.cond.dtype, NativeBool) else expr.cond
        value_true = expr.value_true
        value_false = expr.value_false

        if value_true.dtype != value_false.dtype :
            try :
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
            except KeyError:
                errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
            value_true = cast_func(value_true) if value_true.dtype != expr.dtype else value_true
            value_false = cast_func(value_false) if value_false.dtype != expr.dtype else value_false
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
        if expr.dtype is NativeString():
            return '//'.join('trim('+self._print(a)+')' for a in expr.args)
        else:
            args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
            return ' + '.join(self._print(a) for a in args)

    def _print_PyccelMinus(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        args_code = [self._print(a) for a in args]

        return ' - '.join(args_code)

    def _print_PyccelMul(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        args_code = [self._print(a) for a in args]
        return ' * '.join(a for a in args_code)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype in (NativeBool(), NativeInteger()) for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return ' / '.join(self._print(a) for a in args)

    def _print_PyccelMod(self, expr):
        is_float = expr.dtype is NativeFloat()

        def correct_type_arg(a):
            if is_float and a.dtype is NativeInteger():
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
        adtype   = expr.args[0].dtype
        is_float = expr.dtype is NativeFloat()
        for b in expr.args[1:]:
            bdtype    = b.dtype
            if adtype is NativeInteger() and bdtype is NativeInteger():
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
        if expr.dtype is NativeBool():
            return ' .neqv. '.join(self._print(a) for a in expr.args)
        return 'IEOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is NativeBool():
            return ' .or. '.join(self._print(a) for a in expr.args)
        return 'IOR({}, {})'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is NativeBool():
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
        args = [a if a.dtype is NativeBool() else PythonBool(a) for a in expr.args]
        return ' .and. '.join(self._print(a) for a in args)

    def _print_PyccelOr(self, expr):
        args = [a if a.dtype is NativeBool() else PythonBool(a) for a in expr.args]
        return ' .or. '.join(self._print(a) for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .eqv. {}'.format(lhs, rhs)
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        a = expr.args[0].dtype
        b = expr.args[1].dtype

        if a is NativeBool() and b is NativeBool():
            return '{} .neqv. {}'.format(lhs, rhs)
        return '{0} /= {1}'.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        args = [PythonInt(a) if a.dtype is NativeBool() else a for a in expr.args]
        lhs = self._print(args[0])
        rhs = self._print(args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        if (expr.args[0].dtype is not NativeBool()):
            return '{} == 0'.format(a)
        return '.not. {}'.format(a)

    def _print_Header(self, expr):
        return ''

    def _print_SysExit(self, expr):
        code = ""
        if expr.status.dtype is not NativeInteger() or expr.status.rank > 0:
            print_arg = FunctionCallArgument(expr.status)
            code = self._print(PythonPrint((print_arg, ), file="stderr"))
            arg = "1"
        else:
            arg = expr.status
            if arg.precision != 4:
                arg = NumpyInt32(arg)
            arg = self._print(arg)
        return f'{code}stop {arg}\n'

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_fortran[type_name]
        except KeyError:
            self._print_not_supported(expr)
        args = [self._print(NumpyFloat(a) if a.dtype is NativeInteger() else a)\
				for a in expr.args]
        code_args = ', '.join(args)
        code = '{0}({1})'.format(func_name, code_args)
        return self._get_statement(code)

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
        func = PyccelFunctionDef('numpy_sign', NumpySign)
        self._additional_imports.add(Import('numpy_f90', AsName(func, 'numpy_sign')))
        return f'numpy_sign({self._print(expr.args[0])})'

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
            self._additional_imports.add(Import('pyc_math_f90', Module('pyc_math_f90',(),())))
        args = []
        for arg in expr.args:
            if arg.dtype != expr.dtype:
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
                args.append(self._print(cast_func(arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathCeil(self, expr):
        """Convert a Python expression with a math ceil function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "ceiling({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to
        Fortran function call"""
        # add necessary include
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
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
        if arg.dtype is NativeInteger() or arg.dtype is NativeBool():
            arg = NumpyFloat(arg)
        code_args = self._print(arg)
        code = 'sqrt({})'.format(code_args)
        return self._get_statement(code)

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
        base_code = self._print(base)

        inds = list(expr.indices)
        if expr.base.order == 'C':
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

        inds = [self._print(i) for i in inds]

        return "%s(%s)" % (base_code, ", ".join(inds))

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
        parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign))
        is_function =  len(func_results) == 1 and func_results[0].rank == 0

        if isinstance(expr, DottedFunctionCall):
            args = args[1:]

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
                    results_strs = ['{} = {}'.format(self._print(n), self._print(r))
                            for n,r in lhs_vars.items()]
                else:
                    results_strs = [self._print(r) for r in lhs_vars.values()]

        elif not is_function and len(func_results)!=0:
            results = [r.clone(name = self.scope.get_new_name()) \
                        for r in func_results]
            for var in results:
                self.scope.insert_variable(var)

            results_strs = ['{} = {}'.format(self._print(n), self._print(r)) \
                            for n,r in zip(func_results, results)]

        else:
            results_strs = []

        if func.is_inline:
            if len(func_results)>1:
                code = self._handle_inline_func_call(expr, args, assign_lhs = results)
            else:
                code = self._handle_inline_func_call(expr, args)
        else:
            args_strs = ['{}'.format(self._print(a)) for a in args if not isinstance(a.value, Nil)]
            args_code = ', '.join(args_strs+results_strs)
            code = '{name}({args})'.format( name = f_name,
                                            args = args_code )
            if not is_function:
                code = 'call {}\n'.format(code)

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
            return '{0} = {1}\n'.format(self._print(results[0]), code)
        else:
            return code

#=======================================================================================

    def _print_DottedFunctionCall(self, expr):
        if isinstance(expr.prefix, FunctionCall):
            base = expr.prefix.funcdef.results[0].var
            if (not self._additional_code):
                self._additional_code = ''
            var_name = self.scope.get_new_name()
            var = base.clone(var_name)

            self.scope.insert_variable(var)

            self._additional_code = self._additional_code + self._print(Assign(var,expr.prefix)) + '\n'
            user_node = expr.current_user_node
            expr = DottedFunctionCall(expr.funcdef, expr.args, var)
            expr.set_current_user_node(user_node)
        return self._print_FunctionCall(expr)

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
        """Accepts a string of code or a list of code lines"""
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
            if line in('','\n'):
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


def fcode(expr, filename, assign_to=None, **settings):
    """Converts an expr to a string of Fortran code

    expr : Expr
        A pyccel expression to be converted.
    filename : str
        The name of the file being translated. Used in error printing
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    """

    return FCodePrinter(filename, **settings).doprint(expr, assign_to)
