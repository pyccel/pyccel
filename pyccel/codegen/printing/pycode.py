# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import ast
import warnings

from pyccel.decorators import __all__ as pyccel_decorators

from pyccel.ast.builtins   import PythonMin, PythonMax, PythonType, PythonBool, PythonInt, PythonFloat
from pyccel.ast.builtins   import PythonComplex, DtypePrecisionToCastFunction, PythonTuple, PythonDict
from pyccel.ast.builtin_methods.list_methods import ListAppend
from pyccel.ast.core       import CodeBlock, Import, Assign, FunctionCall, For, AsName, FunctionAddress, If
from pyccel.ast.core       import IfSection, FunctionDef, Module, PyccelFunctionDef, ClassDef
from pyccel.ast.core       import Interface, FunctionDefArgument, FunctionDefResult
from pyccel.ast.datatypes  import HomogeneousTupleType, HomogeneousListType, HomogeneousSetType
from pyccel.ast.datatypes  import VoidType, DictType, InhomogeneousTupleType, PyccelType
from pyccel.ast.datatypes  import FixedSizeNumericType
from pyccel.ast.functionalexpr import FunctionalFor
from pyccel.ast.internals  import PyccelSymbol
from pyccel.ast.literals   import LiteralTrue, LiteralString, LiteralInteger, Nil
from pyccel.ast.low_level_tools import UnpackManagedMemory
from pyccel.ast.numpyext   import numpy_target_swap, numpy_linalg_mod, numpy_random_mod
from pyccel.ast.numpyext   import NumpyArray, NumpyNonZero, NumpyResultType
from pyccel.ast.numpyext   import process_dtype as numpy_process_dtype
from pyccel.ast.numpyext   import NumpyNDArray, NumpyBool
from pyccel.ast.numpytypes import NumpyNumericType, NumpyNDArrayType
from pyccel.ast.type_annotations import VariableTypeAnnotation, SyntacticTypeAnnotation
from pyccel.ast.typingext  import TypingTypeVar, TypingFinal, TypingAnnotation
from pyccel.ast.utilities  import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities  import decorators_mod
from pyccel.ast.variable   import DottedName, Variable, IndexedElement

from pyccel.parser.semantic import magic_method_map

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

#==============================================================================

# Dictionary mapping imported targets to their aliases used internally by pyccel
# This prevents a mismatch between printed imports and function calls
# The keys are modules from which the target is imported
# The values are a dictionary whose keys are object aliases and whose values
# are the names used in pyccel
import_object_swap = {'numpy': numpy_target_swap}
import_target_swap = {
        'numpy' : {'double'     : 'float64',
                   'prod'       : 'product',
                   'empty_like' : 'empty',
                   'zeros_like' : 'zeros',
                   'ones_like'  : 'ones',
                   'max'        : 'amax',
                   'min'        : 'amin',
                   'T'          : 'transpose',
                   'full_like'  : 'full',
                   'absolute'   : 'abs'},
        'numpy.random' : {'random' : 'rand'}
        }
import_source_swap = {
        'omp_lib' : 'pyccel.stdlib.internal.openmp'
        }

class PythonCodePrinter(CodePrinter):
    """
    A printer for printing code in Python.

    A printer to convert Pyccel's AST to strings of Python code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
        The name of the file being pyccelised.
    verbose : int
        The level of verbosity.
    """
    printmethod = "_pycode"
    language = "python"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename, * , verbose):
        errors.set_target(filename)
        super().__init__(verbose)
        self._aliases = {}
        self._ignore_funcs = []
        self._tuple_assigns = []
        self._in_header = False

    def _indent_codestring(self, lines):
        tab = " "*self._default_settings['tabwidth']
        if lines == '':
            return lines
        else:
            # lines ends with \n
            return tab+lines.strip('\n').replace('\n','\n'+tab)+'\n'

    def _format_code(self, lines):
        return lines

    def _find_functional_expr_and_iterables(self, expr):
        """
        Extract the central expression and iterables from a FunctionalFor or GeneratorComprehension.

        Traverse through the loop representing a FunctionalFor or GeneratorComprehension
        to extract the central expression and the different iterable objects.

        Parameters
        ----------
        expr : FunctionalFor
               The loop or generator comprehension to be analyzed.

        Returns
        -------
        body      : TypedAstNode
                    The expression inside the for loops.
        iterables : list of Iterables
                    The iterables over which the for loops iterate.
        """
        dummy_var = expr.index
        iterables = []
        body = expr.loops[-1]
        while not isinstance(body, (Assign, ListAppend)):
            if isinstance(body, If):
                body = body.blocks[0].body.body[0]
            elif isinstance(body, CodeBlock):
                body = list(body.body)
                while isinstance(body[0], FunctionalFor):
                    func_for = body.pop(0)
                    # Replace the temporary assign value with the FunctionalFor expression
                    # so the loop is printed inline
                    for b in body:
                        b.substitute(func_for.lhs, func_for)
                if len(body) > 1:
                    # Ensure all assigns assign to the dummy we are searching for and do not introduce unexpected variables
                    if any(not(isinstance(b, Assign) and b.lhs is dummy_var) for b in body[1:]):
                        raise NotImplementedError("Pyccel has introduced unnecessary statements which it cannot yet disambiguate in the python printer")
                body = body[0]
            elif isinstance(body, For):
                iterables.append(body.iterable)
                body = body.body
            elif isinstance(body, FunctionalFor):
                body, it = self._find_functional_expr_and_iterables(body)
                iterables.extend(it)
            else:
                raise NotImplementedError("Type {} not handled in a FunctionalFor".format(type(body)))
        return body, iterables

    def _get_numpy_name(self, expr):
        """
        Get the name of a NumPy function and ensure it is imported.

        Get the name of a NumPy function from an instance of the class. The
        name is saved in the class by default, however _aliases are checked
        in case the function was imported explicitly with a different name
        (e.g. `from numpy import int32 as i32`). If the name is not found in
        aliases then it is added to the objects imported from NumPy.

        Parameters
        ----------
        expr : PyccelFunction
            A Pyccel node describing a NumPy function.

        Returns
        -------
        str
            The name that should be used in the code.
        """
        if isinstance(expr, type):
            cls = expr
        else:
            cls = type(expr)
        if cls is NumpyBool:
            return 'bool'
        type_name = expr.name
        name = self._aliases.get(cls, type_name)
        if name == type_name and cls not in (PythonBool, PythonInt, PythonFloat, PythonComplex):
            if type_name in numpy_linalg_mod:
                self.add_import(Import('numpy.linalg', [AsName(cls, name)]))
            elif type_name in numpy_random_mod:
                self.add_import(Import('numpy.random', [AsName(cls, name)]))
            else:
                self.add_import(Import('numpy', [AsName(cls, name)]))
        return name

    def _get_type_annotation(self, obj):
        """
        Get the code for the type annotation.

        Get the code for the type annotation of the object passed as argument.

        Parameters
        ----------
        obj : TypedAstNode
            An object for which a type annotation should be printed.

        Returns
        -------
        str
            A string containing the type annotation.
        """
        if isinstance(obj, FunctionDefArgument):
            is_temp_union_name = isinstance(obj.annotation, SyntacticTypeAnnotation) and \
                                 isinstance(obj.annotation.dtype, PyccelSymbol)
            if obj.annotation and not is_temp_union_name and not self._in_header:
                type_annotation = self._print(obj.annotation)
                return f"'{type_annotation}'"
            else:
                if obj.is_vararg:
                    return self._get_type_annotation(obj.var[0])
                if obj.is_kwarg:
                    type_annotation = self._print(obj.var.class_type.value_type)
                    return f"'{type_annotation}'"
                return self._get_type_annotation(obj.var)
        elif isinstance(obj, FunctionDefResult):
            if obj.var is Nil():
                return ''
            else:
                return self._get_type_annotation(obj.var)
        elif isinstance(obj, (Variable, IndexedElement)):
            type_annotation = self._print(obj.class_type)
            if isinstance(obj, Variable):
                if obj.is_alias:
                    self.add_import(Import('typing', [AsName(TypingAnnotation, 'Annotated')]))
                    type_annotation = f'Annotated[{type_annotation}, "alias"]'
                elif obj.on_stack and obj.rank:
                    self.add_import(Import('typing', [AsName(TypingAnnotation, 'Annotated')]))
                    type_annotation = f'Annotated[{type_annotation}, "stack"]'
            return f"'{type_annotation}'"
        elif isinstance(obj, FunctionAddress):
            args = ', '.join(self._get_type_annotation(a).strip("'") for a in obj.arguments)
            res = self._get_type_annotation(obj.results).strip("'")
            return f"'({res})({args})'"
        else:
            raise NotImplementedError(f"Unexpected object of type {type(obj)}")

    def _function_signature(self, func):
        """
        Print the function signature.

        Print the function signature in a .pyi file. This contains arguments,
        result declarations and type annotations.

        Parameters
        ----------
        func : FunctionDef | Interface
            The function whose signature is of interest.

        Returns
        -------
        str
            The code which describes the function signature.
        """
        interface = func.get_direct_user_nodes(lambda x: isinstance(x, Interface))
        if func.is_inline:
            return self._print(func)

        low_level_name = func.name
        name = func.scope.get_python_name(interface[0].name if interface else func.name)
        wrapping = f"@low_level('{low_level_name}')\n"
        self.add_import(Import('pyccel.decorators',
                               [AsName(FunctionDef('low_level', (), ()), 'low_level')]))

        if interface:
            self.add_import(Import('typing', [AsName(FunctionDef('overload', (), ()), 'overload')]))
            overload = '@overload\n'
        else:
            overload = ''

        self.set_scope(func.scope)
        arguments = func.arguments
        arg_code = [self._print(i) for i in arguments]
        if arguments and arguments[0].is_posonly:
            arg_code.insert(next((i for i, a in enumerate(arguments) if not a.is_posonly), len(arg_code)), '/')
        if arguments and any(a.is_kwonly for a in arguments) and all(not a.is_vararg for a in arguments):
            arg_code.insert(next((i for i, a in enumerate(arguments) if a.is_kwonly), len(arg_code)), '*')
        args   = ', '.join(arg_code)
        result = func.results
        body = '...'
        if result:
            res = f' -> {self._get_type_annotation(result.var)}'
        else:
            res = ' -> None'
        dec = self._handle_decorators(func.decorators)
        self.exit_scope()
        return ''.join((dec, wrapping, overload, f"def {name}({args}){res}:\n",
                        self._indent_codestring(body)))

    def _handle_decorators(self, decorators):
        """
        Print decorators for a function.

        Print all decorators for a function in the expected format.

        Parameters
        ----------
        decorators : dict
            A dictionary describing the function templates.

        Returns
        -------
        str
            The code which describes the decorators.
        """
        if len(decorators) == 0:
            return ''
        dec = ''
        for name,f in decorators.items():
            if name in pyccel_decorators:
                self.add_import(Import(DottedName('pyccel.decorators'), [AsName(decorators_mod[name], name)]))
            # TODO - All decorators must be stored in a list
            if not isinstance(f, list):
                f = [f]
            for func in f:
                if isinstance(func, FunctionCall):
                    args = ', '.join(self._print(a) for a in func.args)
                elif func == name:
                    args = ''
                else:
                    args = ', '.join(self._print(LiteralString(a)) for a in func)

                if args:
                    dec += f'@{name}({args})\n'

                else:
                    dec += f'@{name}\n'
        return dec

    def _get_type_var_declarations(self):
        """
        Print the TypeVar declarations.

        Print the TypeVar declarations that exist in the current scope.

        Returns
        -------
        str
            A string containing the code which declares the TypeVar objects.
        """
        type_vars_in_scope = {n:t for n,t in self.scope.symbolic_aliases.items() \
                            if isinstance(t, TypingTypeVar)}
        type_var_constraints = [", ".join(f"'{self._print(ti)}'" for ti in t.type_list) for t in type_vars_in_scope.values()]
        self.add_import(Import('typing', [AsName(TypingTypeVar, 'TypeVar')]))
        return ''.join(f"{n} = TypeVar('{n}', {t})\n" for n,t in zip(type_vars_in_scope, type_var_constraints))

    def get_type_checks(self, arg_code, class_type):
        """
        Get the code to check if an argument has the expected type.

        Get the code to check if an argument has the expected type. This is used in an
        if block to select the right code to run.

        Parameters
        ----------
        arg_code : str
            The code describing the argument being checked.
        class_type : PyccelType
            The expected type.

        Returns
        -------
        list[str]
            A list containing the checks that must be satisfied for the type to be
            considered matching.
        """
        if isinstance(class_type, NumpyNDArrayType):
            ndarray = self._get_numpy_name(NumpyNDArray)
            dtype = self._get_numpy_name(NumpyResultType)
            check_option = []
            check_option.append(f'isinstance({arg_code}, {ndarray})')
            check_option.append(f'{arg_code}.dtype is {dtype}({self._print(class_type.element_type)})')
            check_option.append(f'{arg_code}.ndim == {class_type.rank}')
            if class_type.order:
                check_option.append(f"{arg_code}.flags['{class_type.order}_CONTIGUOUS']")
            return ' and '.join(check_option)
        elif isinstance(class_type, FixedSizeNumericType):
            return f'isinstance({arg_code}, {self._get_numpy_name(DtypePrecisionToCastFunction[class_type])})'
        elif isinstance(class_type, (HomogeneousListType, HomogeneousSetType, HomogeneousTupleType)):
            check_option = []
            check_option.append(f'isinstance({arg_code}, {class_type.name})')
            element_type = class_type.element_type
            tmp_var_code = self._print(self.scope.get_temporary_variable(element_type))
            elem_check = self.get_type_checks(tmp_var_code, element_type)
            check_option.append(f'all({elem_check} for {tmp_var_code} in {arg_code})')
            return ' and '.join(check_option)
        else:
            raise NotImplementedError(f"Can't print a Python interface for type {class_type}")

    #----------------------------------------------------------------------

    def _print_dtype_argument(self, expr, init_dtype):
        """
        Print a dtype argument.

        Print the argument `dtype=X` from the dtype initially provided.

        Parameters
        ----------
        expr : TypedAstNode
            The expression whose datatype is being determined.

        init_dtype : PythonType, PyccelFunctionDef, LiteralString, str
            The actual dtype passed to the NumPy function.

        Returns
        -------
        str
            The code for the dtype argument.
        """
        if init_dtype is None:
            return ''

        if isinstance(init_dtype, (PythonType, NumpyResultType)):
            dtype = self._print(init_dtype)
        elif isinstance(init_dtype, PyccelFunctionDef):
            dtype = self._get_numpy_name(init_dtype.cls_name)
        else:
            dtype = self._print(expr.dtype)
            if isinstance(expr.dtype, NumpyNumericType):
                dtype = self._get_numpy_name(DtypePrecisionToCastFunction[expr.dtype])
        return f"dtype = {dtype}"

    def _print_Header(self, expr):
        return ''

    def _print_tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '({0})'.format(fs)

    def _print_Variable(self, expr):
        if isinstance(expr.class_type, InhomogeneousTupleType):
            elems = ', '.join(self._print(self.scope.collect_tuple_element(v)) for v in expr)
            if len(expr.class_type) < 2:
                elems += ','
            return f'({elems})'
        else:
            return expr.name

    def _print_DottedVariable(self, expr):
        rhs_code = self._print_Variable(expr)
        lhs_code = self._print(expr.lhs)
        return f"{lhs_code}.{rhs_code}"

    def _print_FunctionDefArgument(self, expr):
        if self._in_header:
            name = self._print(self.scope.get_python_name(expr.name))
        else:
            name = self._print(expr.name)
        default = ''

        if expr.is_vararg:
            name = f'*{name}'
        if expr.is_kwarg:
            name = f'**{name}'

        if expr.annotation and not self._in_header:
            type_annotation = f"'{self._print(expr.annotation)}'"
        else:
            type_annotation = self._get_type_annotation(expr)

        if expr.has_default:
            if isinstance(expr.value, FunctionDef):
                default = f' = {self._print(expr.value.name)}'
            else:
                default = f' = {self._print(expr.value)}'

        return f'{name} : {type_annotation}{default}'

    def _print_FunctionCallArgument(self, expr):
        if expr.keyword:
            if expr.keyword.startswith('**'):
                val = expr.value
                assert isinstance(val, PythonDict)
                assert all(isinstance(k, LiteralString) for k in val.keys)
                keys = [k.python_value for k in val.keys]
                args = [self._print(v) for v in val.values]
                return ', '.join(f'{k} = {a}' for k,a in zip(keys, args))
            elif expr.keyword.startswith('*'):
                val = self._print(expr.value)
                if val.startswith('(') and val.endswith(')'):
                    return val[1:-1].strip(',')
                else:
                    return f'*{val}'
            else:
                val = self._print(expr.value)
                return f'{expr.keyword} = {val}'
        else:
            return self._print(expr.value)

    def _print_Idx(self, expr):
        return self._print(expr.name)

    def _print_IndexedElement(self, expr):
        indices = expr.indices
        if isinstance(indices, (tuple, list)):
            # this a fix since when having a[i,j] the generated code is a[(i,j)]
            if len(indices) == 1 and isinstance(indices[0], (tuple, list)):
                indices = indices[0]

            indices = ','.join(self._print(i) for i in indices)
            if len(indices) == 0:
                indices = '()'
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        base = self._print(expr.base)
        return '{base}[{indices}]'.format(base=base, indices=indices)

    def _print_Interface(self, expr):
        # Print each function in the interface
        func_def_code = []
        for func in expr.functions:
            func_def_code.append(self._print(func))

        # Find all the arguments which lead to the same code snippet.
        # In Python the code is often the same for arguments of different types
        bodies : dict[str, list[list[PyccelType]]] = {}
        for f,c in zip(expr.functions, func_def_code):
            # Split functions after declaration to ignore type declaration differences
            b = c.split(':\n',1)[1]
            bodies.setdefault(b, []).append([a.var.class_type for a in f.arguments])

        if len(bodies) == 1:
            return func_def_code[0]
        else:
            bodies_to_print = {}
            imports = {}
            docstrings = set()
            # Collect imports and docstrings from each sub-function
            for b, arg_types in bodies.items():
                lines = b.split('\n')
                import_start = 0
                if lines[0].strip() == '"""':
                    docstr_end = next(i for i,l in enumerate(lines[1:],1) if l.strip() == '"""')
                    docstr = '\n'.join(lines[:docstr_end+1])
                    docstrings.add(docstr)
                    import_start = docstr_end+1
                import_end = next(i for i,l in enumerate(lines[import_start:], import_start) if not (l.strip().startswith('import ') or l.strip().startswith('from ')))
                imports.update({l:None for l in lines[import_start:import_end]})
                new_body = '\n'.join(lines[import_end:])
                bodies_to_print[new_body] = arg_types

            # Group imports together at top of function
            imports_code = '\n'.join(imports.keys())
            # Ensure docstring is printed in docstring position
            assert len(docstrings) <= 1
            docstr = docstrings.pop() if docstrings else ''

            # Add tests to ensure the correct body is called
            arg_names = [a.var.name for a in expr.functions[0].arguments]
            code = ''
            for i, (b, arg_types) in enumerate(bodies_to_print.items()):
                code += '    if ' if i == 0 else '    elif '
                checks = []
                for a_t in arg_types:
                    checks.append(' and '.join(self.get_type_checks(a,t) for a,t in zip(arg_names, a_t)))
                if len(checks) > 1:
                    code += ' or '.join(f'({c})' for c in checks)
                else:
                    code += checks[0]
                code += ':\n'
                code += self._indent_codestring(b)

            header = func_def_code[0].split(':\n',1)[0] + ':'

            return '\n'.join([l for l in (header, docstr, imports_code, code) if l != ''])

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            self.add_import(Import('pyccel.decorators', [AsName(FunctionDef('inline', (), ()), 'inline')]))
            code = ast.unparse(expr.python_ast) + '\n'
            return code

        interface = expr.get_direct_user_nodes(lambda x: isinstance(x, Interface))
        name = self._print(expr.scope.get_python_name(expr.name))

        self.set_scope(expr.scope)
        imports    = ''.join(self._print(i) for i in expr.imports)
        interfaces = ''.join(self._print(i) for i in expr.interfaces if not i.is_argument)
        functions  = [f for f in expr.functions if not any(f in i.functions for i in expr.interfaces)]
        functions  = ''.join(self._print(f) for f in functions)
        body    = self._print(expr.body)
        body    = self._indent_codestring(body)

        arguments = expr.arguments
        arg_code = [self._print(i) for i in arguments]
        if arguments and arguments[0].is_posonly:
            arg_code.insert(next((i for i, a in enumerate(arguments) if not a.is_posonly), len(arg_code)), '/')
        if arguments and any(a.is_kwonly for a in arguments) and all(not a.is_vararg for a in arguments):
            arg_code.insert(next((i for i, a in enumerate(arguments) if a.is_kwonly), len(arg_code)), '*')
        args = ', '.join(arg_code)

        imports    = self._indent_codestring(imports)
        functions  = self._indent_codestring(functions)
        interfaces = self._indent_codestring(interfaces)

        docstring = self._print(expr.docstring) if expr.docstring else ''
        docstring = self._indent_codestring(docstring)

        body = ''.join([docstring, imports, functions, interfaces, body])

        result_annotation = ("-> '" + self._print(expr.results.annotation) + "'") \
                                if expr.results.annotation else ''

        # Put back return removed in semantic stage
        if name.startswith('__i') and ('__'+name[3:]) in magic_method_map.values():
            body += f'    return {expr.arguments[0].name}\n'

        code = (f'def {name}({args}){result_annotation}:\n'
                f'{body}\n')
        dec = self._handle_decorators(expr.decorators)
        code = f'{dec}{code}'
        headers = expr.headers
        if headers:
            headers = self._print(headers)
            code = '{header}\n{code}'.format(header=headers, code=code)

        self.exit_scope()

        return code

    def _print_PyccelFunctionDef(self, expr):
        cls = expr.cls_name
        if cls.__name__.startswith('Numpy'):
            return self._get_numpy_name(cls)
        else:
            return cls.name

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Return(self, expr):

        result_vars = self.scope.collect_all_tuple_elements(expr.expr)

        if expr.expr is None:
            return 'return\n'

        if expr.stmt:
            # Get expressions that should be printed as they are. Assignments to result variables are not
            # printed as the rhs can be inlined
            to_print = [l for l in expr.stmt.body \
                            if not ((isinstance(l, Assign) and isinstance(l.lhs, Variable) and l.lhs in result_vars)
                                     or isinstance(l, UnpackManagedMemory))]
            # Collect all assignments to easily inline the expressions
            assigns = {a.lhs: a.rhs for a in expr.stmt.body if (isinstance(a, Assign) and isinstance(a.lhs, Variable))}
            assigns.update({a.out_ptr: a.managed_object for a in expr.stmt.body if isinstance(a, UnpackManagedMemory)})
            # Print all expressions that are required before the print
            prelude = ''.join(self._print(l) for l in to_print)
        else:
            assigns = {}
            prelude = ''

        def get_return_code(return_var):
            """ Recursive method which replaces any variables in a return statement whose
            definition is known (via the assigns dict) with the definition. A function is
            required to handle the recursivity implied by an unknown depth of inhomogeneous
            tuples.
            """
            if isinstance(return_var.class_type, InhomogeneousTupleType):
                elem_code = [get_return_code(self.scope.collect_tuple_element(elem)) for elem in return_var]
                return_expr = ', '.join(elem_code)
                if len(elem_code) == 1:
                    return_expr += ','
                return f'({return_expr})'
            else:
                return_expr = assigns.get(return_var, return_var)
                return self._print(return_expr)

        return prelude + f'return {get_return_code(expr.expr)}\n'

    def _print_Program(self, expr):
        mod_scope = self.scope
        self.set_scope(expr.scope)
        modules = expr.get_direct_user_nodes(lambda m: isinstance(m, Module))
        assert len(modules) == 1
        module = modules[0]
        imports = ''.join(self._print(i) for i in expr.imports if i.source_module is not module)
        body     = self._print(expr.body)
        imports += ''.join(self._print(i) for i in self._additional_imports.values())

        body = imports+body
        body = self._indent_codestring(body)

        self.exit_scope()
        if mod_scope:
            self.set_scope(mod_scope)
        return ('if __name__ == "__main__":\n'
                '{body}\n').format(body=body)


    def _print_AsName(self, expr):
        target = self._print(expr.local_alias)
        renamed_object = expr.object
        if isinstance(renamed_object, VariableTypeAnnotation):
            return target

        name = self._print(expr.name)
        if isinstance(renamed_object, (FunctionDef, ClassDef)):
            if renamed_object.scope and not getattr(renamed_object, 'is_inline', False):
                name = self._print(renamed_object.scope.get_python_name(expr.name))

        if name == target:
            return name
        else:
            return '{name} as {target}'.format(name = name, target = target)

    def _print_PythonTuple(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        if len(expr.args) == 1:
            args += ','
        return '('+args+')'

    def _print_PythonList(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '['+args+']'

    def _print_PythonSet(self, expr):
        if len(expr.args) == 0:
            return 'set()'
        args = ', '.join(self._print(i) for i in expr.args)
        return '{'+args+'}'

    def _print_PythonDict(self, expr):
        args = ', '.join(f'{self._print(k)}: {self._print(v)}' for k,v in expr)
        return '{'+args+'}'

    def _print_PythonBool(self, expr):
        arg = self._print(expr.arg)
        if expr.rank:
            return f'{arg}.astype(bool)'
        else:
            return f'bool({arg})'

    def _print_PythonInt(self, expr):
        arg = self._print(expr.arg)
        if expr.rank:
            name = self._get_numpy_name(DtypePrecisionToCastFunction[numpy_process_dtype(expr.dtype)])
            return f'{arg}.astype({name})'
        else:
            name = 'int'
            if isinstance(expr.dtype, NumpyNumericType):
                name = self._get_numpy_name(expr)
            return f'{name}({arg})'

    def _print_PythonFloat(self, expr):
        arg = self._print(expr.arg)
        if expr.rank:
            name = self._get_numpy_name(DtypePrecisionToCastFunction[numpy_process_dtype(expr.dtype)])
            return f'{arg}.astype({name})'
        else:
            name = 'float'
            if isinstance(expr.dtype, NumpyNumericType):
                name = self._get_numpy_name(expr)
            return f'{name}({arg})'

    def _print_PythonComplex(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        if expr.is_cast:
            arg = self._print(expr.internal_var)
            if expr.rank:
                return f'{arg}.astype({name})'
            else:
                return f'{name}({arg})'
        else:
            real = self._print(expr.real)
            imag = self._print(expr.imag)
            return f'{name}({real}, {imag})'

    def _print_NumpyComplex(self, expr):
        if isinstance(expr.dtype, NumpyNumericType):
            name = self._get_numpy_name(expr)
        else:
            name = 'complex'
        if expr.is_cast:
            return '{}({})'.format(name, self._print(expr.internal_var))
        else:
            return '{}({}+{}*1j)'.format(name, self._print(expr.real), self._print(expr.imag))

    def _print_VariableIterator(self, expr):
        return self._print(expr.variable)

    def _print_PythonRange(self, expr):
        return 'range({start}, {stop}, {step})'.format(
                start = self._print(expr.start),
                stop  = self._print(expr.stop ),
                step  = self._print(expr.step ))

    def _print_PythonEnumerate(self, expr):
        if expr.start == 0:
            return 'enumerate({elem})'.format(
                    elem = self._print(expr.element))
        else:
            return 'enumerate({elem},{start})'.format(
                    elem = self._print(expr.element),
                    start = self._print(expr.start))

    def _print_PythonMap(self, expr):
        return 'map({func}, {args})'.format(
                func = self._print(expr.func.name),
                args = self._print(expr.func_args))

    def _print_PythonZip(self, expr):
        args = ', '.join(self._print(a) for a in expr.args)
        return f'zip({args})'

    def _print_PythonReal(self, expr):
        if isinstance(expr.internal_var, Variable):
            return '{}.real'.format(self._print(expr.internal_var))
        else:
            return '({}).real'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        if isinstance(expr.internal_var, Variable):
            return '{}.imag'.format(self._print(expr.internal_var))
        else:
            return '({}).imag'.format(self._print(expr.internal_var))

    def _print_PythonConjugate(self, expr):
        if isinstance(expr.internal_var, Variable):
            return '{}.conjugate()'.format(self._print(expr.internal_var))
        else:
            return '({}).conjugate()'.format(self._print(expr.internal_var))

    def _print_PythonPrint(self, expr):
        return 'print({})\n'.format(', '.join(self._print(a) for a in expr.expr))

    def _print_PyccelArrayShapeElement(self, expr):
        arg = expr.arg
        index = expr.index
        arg_code = self._print(arg)
        if isinstance(arg.class_type, (NumpyNDArrayType, HomogeneousTupleType)) or \
                not isinstance(index, LiteralInteger):
            index_code = self._print(index)
            name = self._get_numpy_name(expr)
            return f'{name}({arg_code})[{index_code}]'
        elif index == 0:
            return f'len({arg_code})'
        else:
            raise NotImplementedError("The shape access function seems to be poorly defined.")

    def _print_PythonRound(self, expr):
        arg = self._print(expr.arg)
        if expr.ndigits:
            ndigits = self._print(expr.ndigits)
            return f'round({arg}, {ndigits})'
        else:
            return f'round({arg})'

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        name = self._get_numpy_name(expr)
        return f'{name}({arg})'

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} \n'.format(txt)

    def _print_CommentBlock(self, expr):
        comment_lines = [c.rstrip() for c in expr.comments]
        if comment_lines[0] != '':
            comment_lines.insert(0, '')
            comment_lines[1] = comment_lines[1].lstrip()
        if comment_lines[-1].strip() != '':
            comment_lines.append('')
        txt = '\n'.join(self._print(c) for c in comment_lines)
        return f'"""{txt}"""\n'

    def _print_Assert(self, expr):
        condition = self._print(expr.test)
        return "assert {0}\n".format(condition)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_DottedName(self, expr):
        # A DottedName can only contain LiteralStrings or PyccelSymbols at the printing stage
        return '.'.join(str(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        if func in self._ignore_funcs:
            return ''

        if func.is_imported:
            func_name = self.scope.get_import_alias(func, 'functions')
        elif expr.interface and expr.interface.is_imported:
            func_name = self.scope.get_import_alias(expr.interface, 'functions')
        else:
            func_name = func.scope.get_python_name(func.name)

        # No need to print module init/del functions in Python
        if func.scope.get_python_name(func.name) in ('__init__', '__del__') and \
                func.is_imported and len(func.arguments) == 0:
            return ''

        args = expr.args
        if func.arguments and func.arguments[0].bound_argument:
            func_name = f'{self._print(args[0])}.{func_name}'
            if 'property' in func.decorators:
                return func_name
            args = args[1:]
        args_str = ', '.join(self._print(i) for i in args)
        code = f'{func_name}({args_str})'
        if expr.funcdef.results:
            return code
        else:
            return code+'\n'

    def _print_Import(self, expr):
        mod = expr.source_module
        init_func = None
        free_func = None
        if mod:
            init_func = mod.init_func
            free_func = mod.free_func

        source = self._print(expr.source)

        source = import_source_swap.get(source, source)

        target = [t for t in expr.target if not (isinstance(t.object, Module) or
                  (isinstance(t.object, FunctionDef) and not t.object.is_inline and t.object.scope and
                   t.object.scope.get_python_name(t.object.name) in ('__init__', '__del__')))]
        mod_target = [t for t in expr.target if isinstance(t.object, Module)]

        prefix = ''
        if mod_target:
            if source in pyccel_builtin_import_registry:
                prefix = ''.join(f'import {t.name} as {t.local_alias}\n' for t in mod_target)
            else:
                assert len(mod_target) == 1
                prefix = f'import {source} as {mod_target[0].local_alias}\n'

        if target:
            if source in import_object_swap:
                target = [AsName(import_object_swap[source].get(i.object,i.object), i.local_alias) for i in target]
            if source in import_target_swap:
                # If the source contains multiple names which reference the same object
                # check if the target is referred to by another name in pyccel.
                # Print the name used by pyccel (either the value from import_target_swap
                # or the original name from the import
                target = [AsName(i.object, import_target_swap[source].get(i.local_alias,i.local_alias)) for i in target]

            target = list(dict.fromkeys(target))
            if source in pyccel_builtin_import_registry:
                self._aliases.update((pyccel_builtin_import_registry[source][t.name].cls_name, t.local_alias) \
                                        for t in target if not isinstance(t.object, VariableTypeAnnotation) and \
                                                           t.name != t.local_alias)

            if init_func:
                self._ignore_funcs.append(init_func)
                if init_func.name in self.scope.imports['functions']:
                    self._ignore_funcs.append(self.scope.imports['functions'][init_func.name])
            if free_func:
                self._ignore_funcs.append(free_func)

            target = [self._print(t) for t in target if t.object not in (init_func, free_func)]
            target = ', '.join(target)
            return prefix + f'from {source} import {target}\n'
        else:
            return prefix

    def _print_CodeBlock(self, expr):
        if len(expr.body)==0:
            return 'pass\n'
        else:
            code = ''.join(self._print(c) for c in expr.body)
            return code

    def _print_For(self, expr):
        self.set_scope(expr.scope)
        iterable = self._print(expr.iterable)
        target   = expr.target
        if not isinstance(target,(list, tuple)):
            target = [target]
        target = ','.join(self._print(i) for i in target)
        body   = self._print(expr.body)
        body   = self._indent_codestring(body)
        code   = ('for {0} in {1}:\n'
                '{2}').format(target,iterable,body)

        self.exit_scope()
        return code

    def _print_FunctionalFor(self, expr):
        body, iterators = self._find_functional_expr_and_iterables(expr)
        lhs = self._print(expr.lhs)
        condition = ''
        if isinstance(body, Assign):
            body = self._print(body.rhs)
        else:
            assert isinstance(body, ListAppend)
            body = self._print(body.args[0])

        for_loops = ' '.join(f'for {self._print(idx)} in {self._print(iters)}{" if " + self._print(condition.blocks[0].condition) if condition else ""}'
                             for idx, iters, condition in zip(expr.indices, iterators, expr.conditions))

        if isinstance(expr.class_type, NumpyNDArrayType):
            array = self._get_numpy_name(NumpyArray)
            return f'{lhs} = {array}([{body} {for_loops} {condition}])\n'
        return f'{lhs} = [{body} {for_loops} {condition}]\n'

    def _print_GeneratorComprehension(self, expr):
        body, iterators = self._find_functional_expr_and_iterables(expr)

        rhs = body.rhs
        if isinstance(rhs, (PythonMax, PythonMin)):
            args = rhs.args[0]
            if body.lhs in args:
                args = [a for a in args if a != body.lhs]
                if len(args)==1:
                    rhs = args[0]
                else:
                    rhs = type(body.rhs)(*args)

        body = self._print(rhs)
        for_loops = ' '.join(f'for {self._print(idx)} in {self._print(iters)}{" if " + self._print(condition.blocks[0].condition) if condition else ""}'
                             for idx, iters, condition in zip(expr.indices, iterators, expr.conditions))

        if expr.get_user_nodes(FunctionalFor):
            return '{}({} {})'.format(expr.name, body, for_loops)
        else:
            lhs = self._print(expr.lhs)
            return '{} = {}({} {})\n'.format(lhs, expr.name, body, for_loops)

    def _print_While(self, expr):
        cond = self._print(expr.test)
        self.set_scope(expr.scope)
        body = self._indent_codestring(self._print(expr.body))
        self.exit_scope()
        return 'while {cond}:\n{body}'.format(
                cond = cond,
                body = body)

    def _print_Break(self, expr):
        return 'break\n'

    def _print_Continue(self, expr):
        return 'continue\n'

    def _print_Assign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs

        if isinstance(rhs, FunctionCall) and (rhs.class_type, InhomogeneousTupleType) and isinstance(lhs, PythonTuple):
            # lhs needs packing back into a tuple
            def pack_lhs(lhs, rhs_type_template):
                new_lhs = []
                i = 0
                for elem in rhs_type_template:
                    if isinstance(elem, InhomogeneousTupleType):
                        tuple_elem = pack_lhs(lhs[i:], rhs_type_template[i])
                        new_lhs.append(tuple_elem)
                        i += len(tuple_elem)
                    else:
                        new_lhs.append(lhs[i])
                        i += 1
                return PythonTuple(*new_lhs)
            lhs = pack_lhs(lhs.args, rhs.class_type)

        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        if isinstance(rhs, Variable) and rhs.rank>1 and rhs.order != lhs.order:
            code = f'{lhs_code} = {rhs_code}.T\n'
        else:
            code = f'{lhs_code} = {rhs_code}\n'

        if isinstance(lhs, IndexedElement) and isinstance(lhs.base.class_type, HomogeneousTupleType):
            assert len(lhs.indices) == 1
            idx = lhs.indices[0]
            self._tuple_assigns.append(code)
            if int(idx) < int(lhs.base.shape[0])-1:
                return ''
            else:
                exprs = self._tuple_assigns
                rhs_elems = ', '.join(e.split(' = ')[1].strip('\n') for e in exprs)
                self._tuple_assigns = []
                if len(exprs) < 2:
                    rhs_elems += ','
                lhs_code = self._print(lhs.base)
                return f'{lhs_code} = ({rhs_elems})\n'
        else:
            return code

    def _print_AliasAssign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs

        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        if isinstance(rhs, Variable) and rhs.order!= lhs.order:
            return'{0} = {1}.T\n'.format(lhs_code,rhs_code)
        else:
            return'{0} = {1}\n'.format(lhs_code,rhs_code)

    def _print_AugAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        op  = self._print(expr.op)
        return'{0} {1}= {2}\n'.format(lhs,op,rhs)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return f'range({start}, {stop}, {step})'

    def _print_Allocate(self, expr):
        class_type = expr.variable.class_type
        if isinstance(class_type, HomogeneousTupleType) and expr.shape[0] == 0:
            var = self._print(expr.variable)
            return f'{var} = ()\n'
        if expr.alloc_type == 'reserve':
            var = self._print(expr.variable)
            if isinstance(class_type, HomogeneousSetType):
                return f'{var} = set()\n'
            elif isinstance(class_type, HomogeneousListType):
                return f'{var} = []\n'
            elif isinstance(class_type, DictType):
                return f'{var} = {{}}\n'

        return ''

    def _print_Deallocate(self, expr):
        return ''

    def _print_NumpyArray(self, expr):
        name = self._get_numpy_name(expr)
        arg_var = expr.arg

        arg   = self._print(arg_var)
        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        order = f"order='{expr.order}'" if expr.order else ''
        ndmin = f"ndmin={expr.rank}" if expr.rank > arg_var.rank else ''
        args  = ', '.join(a for a in [arg, dtype, order, ndmin] if a!= '')
        return f"{name}({args})"

    def _print_NumpyAutoFill(self, expr):
        func_name = self._get_numpy_name(expr)

        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        shape = self._print(expr.shape)
        order = f"order='{expr.order}'" if expr.order else ''
        args  = ', '.join(a for a in [shape, dtype, order] if a!='')
        return f"{func_name}({args})"

    def _print_NumpyLinspace(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        start = self._print(expr.start)
        stop = self._print(expr.stop)
        num = "num = " + self._print(expr.num)
        endpoint = "endpoint = "+self._print(expr.endpoint)
        args = ', '.join(a for a in [start, stop, num, endpoint, dtype] if a != '')
        return f"{name}({args})"

    def _print_NumpyMatmul(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        return "{0}({1}, {2})".format(
                name,
                self._print(expr.a),
                self._print(expr.b))


    def _print_NumpyFull(self, expr):
        name = self._aliases.get(type(expr), expr.name)

        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        shape      = self._print(expr.shape)
        fill_value = self._print(expr.fill_value)
        order      = f"order='{expr.order}'" if expr.order else ''
        args       = ', '.join(a for a in [shape, fill_value, dtype, order] if a)
        return f"{name}({args})"

    def _print_NumpyArange(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        args = ', '.join(a for a in [self._print(expr.start),
                          self._print(expr.stop),
                          self._print(expr.step),
                          dtype] if a != '')
        return f"{name}({args})"

    def _print_PyccelFunction(self, expr):
        name = self._aliases.get(type(expr),expr.name)
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_NumpyResultType(self, expr):
        args = expr.args
        if len(args) == 1 and args[0].rank > 1:
            arg = args[0]
            arg_code = self._print(arg)
            return f"{arg_code}.dtype"
        else:
            name = self._get_numpy_name(expr)
            args = ', '.join(self._print(a) for a in expr.args)
            return f"{name}({args})"

    def _print_NumpyRandint(self, expr):
        name = self._get_numpy_name(expr)
        args = []
        if expr.low:
            args.append(self._print(expr.low))
        args.append(self._print(expr.high))
        if expr.rank != 0:
            size = self._print(expr.shape)
            args.append(f"size = {size}")
        return f"{name}({', '.join(args)})"

    def _print_NumpyNorm(self, expr):
        name = self._get_numpy_name(expr)
        axis = self._print(expr.axis) if expr.axis else None
        if axis:
            return  "{name}({arg},axis={axis})".format(name = name, arg  = self._print(expr.python_arg), axis=axis)
        return  "{name}({arg})".format(name = name, arg  = self._print(expr.python_arg))

    def _print_NumpyNonZero(self, expr):
        name = self._aliases.get(type(expr),'nonzero')
        if name == 'nonzero':
            self.add_import(Import('numpy', [AsName(NumpyNonZero, 'nonzero')]))
        arg = self._print(expr.array)
        return "{}({})".format(name, arg)

    def _print_NumpyCountNonZero(self, expr):
        name = self._aliases.get(type(expr),'count_nonzero')
        if name == 'count_nonzero':
            self.add_import(Import('numpy', [AsName(NumpyNonZero, 'count_nonzero')]))

        axis_arg = expr.axis

        arr = self._print(expr.array)
        axis = '' if axis_arg is None else (self._print(axis_arg) + ', ')
        keep_dims = 'keepdims = {}'.format(self._print(expr.keep_dims))

        arg = '{}, {}{}'.format(arr, axis, keep_dims)

        return "{}({})".format(name, arg)

    def _print_NumpyDivide(self, expr):
        args = ', '.join(self._print(a) for a in expr.args)
        name = self._get_numpy_name(type(expr))
        return f'{name}({args})'

    def _print_ListMethod(self, expr):
        method_name = expr.name
        list_obj = self._print(expr.list_obj)
        if len(expr.args) == 0 or all(arg is None for arg in expr.args):
            method_args = ''
        else:
            method_args = ', '.join(self._print(a) for a in expr.args)

        code = f"{list_obj}.{method_name}({method_args})"
        if isinstance(expr.class_type, VoidType):
            return code + '\n'
        else:
            return code

    def _print_DictMethod(self, expr):
        method_name = expr.name
        dict_obj = self._print(expr.dict_obj)
        method_args = ', '.join(self._print(a) for a in expr.args)

        code = f"{dict_obj}.{method_name}({method_args})"
        if isinstance(expr.class_type, VoidType):
            return f'{code}\n'
        else:
            return code

    def _print_DictPop(self, expr):
        dict_obj = self._print(expr.dict_obj)
        key = self._print(expr.key)
        if expr.default_value:
            val = self._print(expr.default_value)
            return f"{dict_obj}.pop({key}, {val})"
        else:
            return f"{dict_obj}.pop({key})"

    def _print_DictGet(self, expr):
        dict_obj = self._print(expr.dict_obj)
        key = self._print(expr.key)
        if expr.default_value:
            val = self._print(expr.default_value)
            return f"{dict_obj}.get({key}, {val})"
        else:
            return f"{dict_obj}.get({key})"

    def _print_DictItems(self, expr):
        dict_obj = self._print(expr.variable)

        return f"{dict_obj}.items()"

    def _print_DictKeys(self, expr):
        dict_obj = self._print(expr.variable)

        return f"{dict_obj}.keys()"

    def _print_DictValues(self, expr):
        dict_obj = self._print(expr.variable)

        return f"{dict_obj}.values()"

    def _print_DictGetItem(self, expr):
        dict_obj = self._print(expr.dict_obj)
        key = self._print(expr.key)
        return f"{dict_obj}[{key}]"

    def _print_Slice(self, expr):
        start = self._print(expr.start) if expr.start else ''
        stop  = self._print(expr.stop)  if expr.stop  else ''
        step  = self._print(expr.step)  if expr.step  else ''
        return '{start}:{stop}:{step}'.format(
                start = start,
                stop  = stop,
                step  = step)

    def _print_LiteralEllipsis(self, expr):
        return '...'

    def _print_SetMethod(self, expr):
        set_var = self._print(expr.set_obj)
        name = expr.name
        args = "" if len(expr.args) == 0 or expr.args[-1] is None \
            else ', '.join(self._print(a) for a in expr.args)
        code = f"{set_var}.{name}({args})"
        if expr.class_type is VoidType():
            return f'{code}\n'
        else:
            return code

    def _print_Nil(self, expr):
        return 'None'

    def _print_Pass(self, expr):
        return 'pass\n'

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is {1}'.format(lhs,rhs)

    def _print_PyccelIsNot(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is not {1}'.format(lhs,rhs)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            if i == 0:
                lines.append("if %s:\n" % self._print(c))

            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else:\n")

            else:
                lines.append("elif %s:\n" % self._print(c))

            if isinstance(e, CodeBlock):
                body = self._indent_codestring(self._print(e))
                lines.append(body)
            else:
                lines.append(self._print(e))
        return "".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{true} if {cond} else {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_Literal(self, expr):
        dtype = expr.dtype

        if isinstance(dtype, NumpyNumericType):
            cast_func = DtypePrecisionToCastFunction[dtype]
            type_name = cast_func.__name__.lower()
            is_numpy  = type_name.startswith('numpy')
            cast_name = cast_func.name
            name = self._aliases.get(cast_func, cast_name)
            if is_numpy and name == cast_name:
                self.add_import(Import('numpy', [AsName(cast_func, cast_name)]))
            return '{}({})'.format(name, repr(expr.python_value))
        else:
            return repr(expr.python_value)

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))

            elif isinstance(f, tuple):
                for i in f:
                    args.append(self._print(i))

            else:
                args.append(self._print(f))

        fs = ', '.join(i for i in args)

        return 'print({0})\n'.format(fs)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)

        type_var_declarations = self._get_type_var_declarations()

        # Insert existing imports so new imports don't cause duplicates
        for i in expr.imports:
            self.add_import(i)
            source = i.source
            if source in pyccel_builtin_import_registry:
                self._aliases.update((pyccel_builtin_import_registry[source][t.name].cls_name, t.local_alias) \
                                        for t in i.target if not isinstance(t.object, (Module, VariableTypeAnnotation)) and \
                                                           t.name != t.local_alias)

        imports = ''.join(self._print(i) for i in expr.imports)

        # Print interface functions (one function with multiple decorators describes the problem)
        interfaces = ''.join(self._print(i) for i in expr.interfaces)
        # Collect functions which are not in an interface
        funcs = [f for f in expr.funcs if not (any(f in i.functions for i in expr.interfaces) \
                        or f is expr.init_func or f is expr.free_func)]
        funcs = ''.join(self._print(f) for f in funcs)
        classes = ''.join(self._print(c) for c in expr.classes)

        init_func = expr.init_func
        if init_func:
            self._ignore_funcs.append(init_func)
            # Collect initialisation body
            init_if = init_func.get_attribute_nodes(IfSection)[0]
            # Remove boolean from init_body
            init_body = init_if.body.body[:-1]
            init_body = ''.join(self._print(l) for l in init_body)
        else:
            init_body = ''

        free_func = expr.free_func
        if free_func:
            self._ignore_funcs.append(free_func)

        imports = ''.join(self._print(i) for i in self._additional_imports.values())

        body = '\n'.join((type_var_declarations, interfaces, funcs, classes, init_body))

        if expr.program:
            expr.program.remove_import(expr.name)
            prog = self._print(expr.program)
        else:
            prog = ''

        self.exit_scope()
        return ('{imports}\n'
                '{body}'
                '{prog}').format(
                        imports = imports,
                        body    = body,
                        prog    = prog)

    def _print_ModuleHeader(self, expr):
        self._in_header = True
        mod = expr.module
        variables = mod.variables

        self.set_scope(mod.scope)
        type_var_declarations = self._get_type_var_declarations()

        # Insert existing imports so new imports don't cause duplicates
        for i in mod.imports:
            self.add_import(i)
        init_func = mod.init_func
        var_decl = ''.join(f"{mod.scope.get_python_name(v.name)} : {self._get_type_annotation(v)}\n"
                            for v in variables if not v.is_temp)
        funcs = ''.join(f'{self._function_signature(f)}\n' for f in mod.funcs)
        funcs += ''.join(f'{self._function_signature(f)}\n' for i in mod.interfaces for f in i.functions)
        classes = ''
        for classDef in mod.classes:
            ll_name = classDef.name
            py_name = classDef.scope.get_python_name(ll_name)
            classes += f"@low_level('{ll_name}')\n"
            classes += f"class {py_name}:\n"
            class_body  = '\n'.join(f"{classDef.scope.get_python_name(v.name)} : {self._get_type_annotation(v)}"
                                    for v in classDef.attributes) + '\n\n'
            for method in classDef.methods:
                class_body += f"{self._function_signature(method)}\n"
            for interface in classDef.interfaces:
                for method in interface.functions:
                    class_body += f"{self._function_signature(method)}\n"

            classes += self._indent_codestring(class_body)

        imports = ''.join(self._print(i) for i in self._additional_imports.values())

        self.exit_scope()

        self._in_header = False

        return '\n'.join(section for section in (imports, type_var_declarations, var_decl, classes, funcs)
                         if section)

    def _print_AllDeclaration(self, expr):
        values = ',\n           '.join(self._print(v) for v in expr.values)
        return f'__all__ = ({values},)\n'

    def _print_PyccelPow(self, expr):
        base = self._print(expr.args[0])
        e    = self._print(expr.args[1])
        return '{} ** {}'.format(base, e)

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        return ' - '.join(self._print(a) for a in expr.args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        return ' / '.join(self._print(a) for a in expr.args)

    def _print_PyccelMod(self, expr):
        return '%'.join(self._print(a) for a in expr.args)

    def _print_PyccelFloorDiv(self, expr):
        return '//'.join(self._print(a) for a in expr.args)

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        return ' and '.join(self._print(a) for a in expr.args)

    def _print_PyccelOr(self, expr):
        return ' or '.join(self._print(a) for a in expr.args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1} '.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1} '.format(lhs, rhs)

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
        return 'not {}'.format(a)

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelRShift(self, expr):
        return '{} >> {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return '{} << {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        return '{} ^ {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        return '{} | {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        return '{} & {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_Duplicate(self, expr):
        return '{} * {}'.format(self._print(expr.val), self._print(expr.length))

    def _print_Concatenate(self, expr):
        return ' + '.join([self._print(a) for a in expr.args])

    def _print_PyccelIn(self, expr):
        element = self._print(expr.element)
        container = self._print(expr.container)
        return f'{element} in {container}'

    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_PythonType(self, expr):
        return 'type({})'.format(self._print(expr.arg))

    def _print_UnpackManagedMemory(self, expr):
        lhs = self._print(expr.out_ptr)
        rhs = self._print(expr.managed_object)
        return f'{lhs} = {rhs}\n'

    #-----------------Class Printer---------------------------------

    def _print_ClassDef(self, expr):
        name = self.scope.get_python_name(expr.name)
        superclasses = ', '.join(self._print(arg) for arg in  expr.superclasses)
        classDefName = f'class {name}({superclasses}):'
        docstring = self._indent_codestring(self._print(expr.docstring)) if expr.docstring else ''
        methods = ''.join(self._print(method) for method in expr.methods)
        methods = self._indent_codestring(methods)
        interfaces = ''.join(self._print(method) for method in expr.interfaces)
        interfaces = self._indent_codestring(interfaces)
        classDef = '\n'.join([classDefName, docstring, methods, interfaces]) + '\n'
        return classDef

    def _print_ConstructorCall(self, expr):
        cls_variable = expr.cls_variable
        try:
            cls_name = self.scope.get_import_alias(cls_variable.class_type, 'cls_constructs')
        except RuntimeError:
            cls_name = cls_variable.cls_base.name
        args = ', '.join(self._print(arg) for arg in expr.args[1:])
        if expr.get_direct_user_nodes(lambda u: isinstance(u, CodeBlock)):
            return f"{cls_variable} = {cls_name}({args})\n"
        else:
            return f"{cls_name}({args})"

    def _print_Del(self, expr):
        return ''.join(f'del {var.variable}\n' for var in expr.variables)

    #------------------OmpAnnotatedComment Printer------------------

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            clauses = ' ' + expr.combined

        omp_expr = '#$omp {}'.format(expr.name)
        clauses += str(expr.txt)
        omp_expr = '{}{}\n'.format(omp_expr, clauses)

        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        omp_expr = str(expr.txt)
        omp_expr = '#$omp {}\n'.format(omp_expr)
        return omp_expr

    #------------------Annotation Printer------------------

    def _print_UnionTypeAnnotation(self, expr):
        types = [self._print(t) for t in expr.type_list]
        return ' | '.join(types)

    def _print_SyntacticTypeAnnotation(self, expr):
        dtype = self._print(expr.dtype)
        dtype = dtype.replace('::',':')
        order = f"(order={expr.order})" if expr.order else ''
        return f'{dtype}{order}'

    def _print_FunctionTypeAnnotation(self, expr):
        args = ', '.join(self._print(a.annotation) for a in expr.args)
        if expr.result.annotation:
            results = self._print(expr.result.annotation)
        else:
            results = ''
        return f"({results})({args})"

    def _print_VariableTypeAnnotation(self, expr):
        return self._print(expr.class_type)

    def _print_FinalType(self, expr):
        annotation = self._print(expr.underlying_type)
        self.add_import(Import('typing', [AsName(TypingFinal, 'Final')]))
        return f'Final[{annotation}]'

    def _print_NumpyNDArrayType(self, expr):
        dims = ','.join(':'*expr.container_rank)
        order_str = f'(order={expr.order})' if expr.order else ''
        return f'{self._print(expr.element_type)}[{dims}]{order_str}'

    def _print_InhomogeneousTupleType(self, expr):
        args = ', '.join(self._print(t) for t in expr)
        if args:
            return f'tuple[{args}]'
        else:
            return 'tuple[()]'

    def _print_HomogeneousTupleType(self, expr):
        return f'tuple[{self._print(expr.element_type)}, ...]'

    def _print_HomogeneousListType(self, expr):
        return f'list[{self._print(expr.element_type)}]'

    def _print_HomogeneousSetType(self, expr):
        return f'set[{self._print(expr.element_type)}]'

    def _print_DictType(self, expr):
        return f'dict[{self._print(expr.key_type)}, {self._print(expr.value_type)}]'

    def _print_PythonNativeBool(self, expr):
        return 'bool'

    def _print_PythonNativeInt(self, expr):
        return 'int'

    def _print_PythonNativeFloat(self, expr):
        return 'float'

    def _print_PythonNativeComplex(self, expr):
        return 'complex'

    def _print_StringType(self, expr):
        return 'str'

    def _print_CustomDataType(self, expr):
        try:
            name = self.scope.get_import_alias(expr, 'cls_constructs')
        except RuntimeError:
            name = expr.name
        return name

    def _print_NumpyNumericType(self, expr):
        name = str(expr).removeprefix('numpy.')
        self.add_import(Import('numpy', [AsName(VariableTypeAnnotation(expr), name)]))
        return name
