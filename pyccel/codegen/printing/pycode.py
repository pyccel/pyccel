# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
import warnings

from pyccel.decorators import __all__ as pyccel_decorators

from pyccel.ast.builtins   import PythonMin, PythonMax, PythonType
from pyccel.ast.core       import CodeBlock, Import, Assign, FunctionCall, For, AsName, FunctionAddress
from pyccel.ast.core       import IfSection, FunctionDef, Module, DottedFunctionCall, PyccelFunctionDef
from pyccel.ast.datatypes  import default_precision, datatype
from pyccel.ast.functionalexpr import FunctionalFor
from pyccel.ast.literals   import LiteralTrue, LiteralString
from pyccel.ast.literals   import LiteralInteger, LiteralFloat, LiteralComplex
from pyccel.ast.numpyext   import NumpyShape, NumpySize, numpy_target_swap
from pyccel.ast.numpyext   import NumpyArray, NumpyNonZero, NumpyResultType
from pyccel.ast.numpyext   import DtypePrecisionToCastFunction
from pyccel.ast.variable   import DottedName, HomogeneousTupleVariable, Variable
from pyccel.ast.utilities  import builtin_import_registry as pyccel_builtin_import_registry
from pyccel.ast.utilities  import decorators_mod

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
import_object_swap = { 'numpy': numpy_target_swap}
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
    """
    printmethod = "_pycode"
    language = "python"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename):
        errors.set_target(filename, 'file')
        super().__init__()
        self._additional_imports = {}
        self._aliases = {}
        self._ignore_funcs = []

    def _indent_codestring(self, lines):
        tab = " "*self._default_settings['tabwidth']
        if lines == '':
            return lines
        else:
            # lines ends with \n
            return tab+lines.strip('\n').replace('\n','\n'+tab)+'\n'

    def _format_code(self, lines):
        return lines

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        imports = [i for tup in self._additional_imports.values() for i in tup[1]]
        return imports

    def insert_new_import(self, source, target, alias = None):
        """ Add an import of an object which may have been
        added by pyccel and therefore may not have been imported
        """
        if alias and alias!=target:
            target = AsName(target, alias)
        import_obj = Import(source, target)
        source = str(source)
        src_info = self._additional_imports.setdefault(source, (set(), []))
        if any(i not in src_info[0] for i in import_obj.target):
            src_info[0].update(import_obj.target)
            src_info[1].append(import_obj)

    def _find_functional_expr_and_iterables(self, expr):
        """
        Traverse through the loop representing a FunctionalFor or GeneratorComprehension
        to extract the central expression and the different iterable objects

        Parameters
        ----------
        expr : FunctionalFor

        Returns
        -------
        body      : PyccelAstNode
                    The expression inside the for loops
        iterables : list of Iterables
                    The iterables over which the for loops iterate
        """
        dummy_var = expr.index
        iterables = []
        body = expr.loops[1]
        while not isinstance(body, Assign):
            if isinstance(body, CodeBlock):
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
        expr : PyccelInternalFunction
            A pyccel node describing a NumPy function.

        Returns
        -------
        str
            The name that should be used in the code.
        """
        if isinstance(expr, type):
            cls = expr
        else:
            cls = type(expr)
        type_name = expr.name
        name = self._aliases.get(cls, type_name)
        if name == type_name:
            self.insert_new_import(
                    source = 'numpy',
                    target = AsName(cls, name))
        return name

    #----------------------------------------------------------------------

    def _print_dtype_argument(self, expr, init_dtype):
        if init_dtype is None:
            return ''
        elif isinstance(init_dtype, (PythonType, NumpyResultType)):
            dtype = self._print(init_dtype)
            return "dtype = " + dtype
        elif isinstance(init_dtype, PyccelFunctionDef):
            dtype = self._get_numpy_name(init_dtype.cls_name)
            return "dtype={}".format(dtype)
        else:
            dtype = self._print(expr.dtype)
            if expr.precision != -1:
                dtype = self._get_numpy_name(DtypePrecisionToCastFunction[datatype(dtype).name][expr.precision])
            return "dtype={}".format(dtype)

    def _print_Header(self, expr):
        return ''

    def _print_tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '({0})'.format(fs)

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeFloat(self, expr):
        return 'float'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_DottedVariable(self, expr):
        rhs_code = self._print_Variable(expr)
        lhs_code = self._print(expr.lhs)
        return f"{lhs_code}.{rhs_code}"

    def _print_FunctionDefArgument(self, expr):
        name = self._print(expr.name)
        type_annotation = ''
        default = ''

        if expr.annotation:
            type_annotation = f' : {expr.annotation}'

        if expr.has_default:
            if isinstance(expr.value, FunctionDef):
                default = f' = {self._print(expr.value.name)}'
            else:
                default = f' = {self._print(expr.value)}'

        return f'{name}{type_annotation}{default}'

    def _print_FunctionCallArgument(self, expr):
        if expr.keyword:
            return '{} = {}'.format(expr.keyword, self._print(expr.value))
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

            indices = [self._print(i) for i in indices]
            if isinstance(expr.base, HomogeneousTupleVariable):
                indices = ']['.join(i for i in indices)
            else:
                indices = ','.join(i for i in indices)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        base = self._print(expr.base)
        return '{base}[{indices}]'.format(base=base, indices=indices)

    def _print_Interface(self, expr):
        # TODO: Improve. See #885
        func = expr.functions[0]
        if not isinstance(func, FunctionAddress):
            func.rename(expr.name)
        func_code = self._print(func)
        _, body = func_code.split(':\n',1)
        for func in expr.functions[1:]:
            if not isinstance(func, FunctionAddress):
                func.rename(expr.name)
            i_func_code = self._print(func)
            _, i_body = i_func_code.split(':\n',1)
            if i_body != body:
                warnings.warn(UserWarning("Generated code varies between interfaces but has not been printed. This Python code may produce unexpected results."))
                return func_code
        return func_code

    def _print_FunctionDef(self, expr):
        self.set_scope(expr.scope)
        name       = self._print(expr.name)
        imports    = ''.join(self._print(i) for i in expr.imports)
        interfaces = ''.join(self._print(i) for i in expr.interfaces if not i.is_argument)
        functions  = [f for f in expr.functions if not any(f in i.functions for i in expr.interfaces)]
        functions  = ''.join(self._print(f) for f in functions)
        body    = self._print(expr.body)
        body    = self._indent_codestring(body)
        args    = ', '.join(self._print(i) for i in expr.arguments)

        imports    = self._indent_codestring(imports)
        functions  = self._indent_codestring(functions)
        interfaces = self._indent_codestring(interfaces)

        doc_string = self._print(expr.doc_string) if expr.doc_string else ''
        doc_string = self._indent_codestring(doc_string)

        body = ''.join([doc_string, functions, interfaces, imports, body])

        code = ('def {name}({args}):\n'
                '{body}\n').format(
                        name=name,
                        args=args,
                        body=body)
        decorators = expr.decorators.copy()
        if decorators:
            if decorators['template']:
                # Eliminate template_dict because it is useless in the printing
                decorators['template'] = decorators['template']['decorator_list']
            else:
                decorators.pop('template')
            for n,f in decorators.items():
                if n in pyccel_decorators:
                    self.insert_new_import(DottedName('pyccel.decorators'), AsName(decorators_mod[n], n))
                # TODO - All decorators must be stored in a list
                if not isinstance(f, list):
                    f = [f]
                dec = ''
                for func in f:
                    if isinstance(func, FunctionCall):
                        args = func.args
                    elif func == n:
                        args = []
                    else:
                        args = [LiteralString(a) for a in func]
                    if n == 'types' and len(args)==0:
                        continue
                    if args:
                        args = ', '.join(self._print(i) for i in args)
                        dec += '@{name}({args})\n'.format(name=n, args=args)

                    else:
                        dec += '@{name}\n'.format(name=n)

                code = '{dec}{code}'.format(dec=dec, code=code)
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

        if expr.stmt:
            assigns = {i.lhs: i.rhs for i in expr.stmt.body if isinstance(i, Assign)}
            prelude = ''.join([self._print(i) for i in expr.stmt.body if not isinstance(i, Assign)])
        else:
            assigns = {}
            prelude = ''
        expr_return_vars = [assigns.get(a,a) for a in expr.expr]

        return prelude+'return {}\n'.format(','.join(self._print(i) for i in expr_return_vars))

    def _print_Program(self, expr):
        mod_scope = self.scope
        self.set_scope(expr.scope)
        imports  = ''.join(self._print(i) for i in expr.imports)
        body     = self._print(expr.body)
        imports += ''.join(self._print(i) for i in self.get_additional_imports())

        body = imports+body
        body = self._indent_codestring(body)

        self.exit_scope()
        if mod_scope:
            self.set_scope(mod_scope)
        return ('if __name__ == "__main__":\n'
                '{body}\n').format(body=body)


    def _print_AsName(self, expr):
        name = self._print(expr.name)
        target = self._print(expr.target)
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

    def _print_PythonBool(self, expr):
        return 'bool({})'.format(self._print(expr.arg))

    def _print_PythonInt(self, expr):
        name = 'int'
        if expr.precision != -1:
            name = self._get_numpy_name(expr)
        return '{}({})'.format(name, self._print(expr.arg))

    def _print_PythonFloat(self, expr):
        name = 'float'
        if expr.precision != -1:
            name = self._get_numpy_name(expr)
        return '{}({})'.format(name, self._print(expr.arg))

    def _print_PythonComplex(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        if expr.is_cast:
            return '{}({})'.format(name, self._print(expr.internal_var))
        else:
            return '{}({}, {})'.format(name, self._print(expr.real), self._print(expr.imag))

    def _print_NumpyComplex(self, expr):
        if expr.precision != -1:
            name = self._get_numpy_name(expr)
        else:
            name = 'complex'
        if expr.is_cast:
            return '{}({})'.format(name, self._print(expr.internal_var))
        else:
            return '{}({}+{}*1j)'.format(name, self._print(expr.real), self._print(expr.imag))

    def _print_Iterable(self, expr):
        return self._print(expr.iterable)

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

    def _print_PythonReal(self, expr):
        return '({}).real'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return '({}).imag'.format(self._print(expr.internal_var))

    def _print_PythonConjugate(self, expr):
        return '({}).conjugate()'.format(self._print(expr.internal_var))

    def _print_PythonPrint(self, expr):
        return 'print({})\n'.format(', '.join(self._print(a) for a in expr.expr))

    def _print_PyccelArrayShapeElement(self, expr):
        arg = self._print(expr.arg)
        index = self._print(expr.index)
        name = self._get_numpy_name(expr)
        return f'{name}({arg})[{index}]'

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        name = self._get_numpy_name(expr)
        return f'{name}({arg})'

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} \n'.format(txt)

    def _print_CommentBlock(self, expr):
        txt = '\n'.join(self._print(c) for c in expr.comments)
        return '"""{0}"""\n'.format(txt)

    def _print_Assert(self, expr):
        condition = self._print(expr.test)
        return "assert {0}\n".format(condition)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_DottedName(self, expr):
        return '.'.join(self._print(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        if expr.funcdef in self._ignore_funcs:
            return ''
        if expr.interface:
            func_name = expr.interface_name
        else:
            func_name = expr.func_name
        args = expr.args
        if isinstance(expr, DottedFunctionCall):
            args = args[1:]
        args_str = ', '.join(self._print(i) for i in args)
        code = f'{func_name}({args_str})'
        if expr.funcdef.results:
            return code
        else:
            return code+'\n'

    def _print_Import(self, expr):
        mod = expr.source_module
        init_func_name = ''
        free_func_name = ''
        if mod:
            init_func = mod.init_func
            if init_func:
                init_func_name = init_func.name
            free_func = mod.free_func
            if free_func:
                free_func_name = free_func.name

        if isinstance(expr.source, AsName):
            source = self._print(expr.source.name)
        else:
            source = self._print(expr.source)

        source = import_source_swap.get(source, source)

        target = [t for t in expr.target if not isinstance(t.object, Module)]

        if not target:
            return 'import {source}\n'.format(source=source)
        else:
            if source in import_object_swap:
                target = [AsName(import_object_swap[source].get(i.object,i.object), i.target) for i in target]
            if source in import_target_swap:
                # If the source contains multiple names which reference the same object
                # check if the target is referred to by another name in pyccel.
                # Print the name used by pyccel (either the value from import_target_swap
                # or the original name from the import
                target = [AsName(i.object, import_target_swap[source].get(i.target,i.target)) for i in target]

            target = list(set(target))
            if source in pyccel_builtin_import_registry:
                self._aliases.update([(pyccel_builtin_import_registry[source][t.name].cls_name, t.target) for t in target if t.name != t.target])

            if expr.source_module:
                if expr.source_module.init_func:
                    self._ignore_funcs.append(expr.source_module.init_func)
                if expr.source_module.free_func:
                    self._ignore_funcs.append(expr.source_module.free_func)
            target = [self._print(t) for t in target if t.name not in (init_func_name, free_func_name)]
            target = ', '.join(target)
            return 'from {source} import {target}\n'.format(source=source, target=target)

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
        body = self._print(body.rhs)
        for_loops = ' '.join(['for {} in {}'.format(self._print(idx), self._print(iters))
                        for idx, iters in zip(expr.indices, iterators)])

        name = self._aliases.get(type(expr),'array')
        if name == 'array':
            self.insert_new_import(
                    source = 'numpy',
                    target = AsName(NumpyArray, 'array'))

        return '{} = {}([{} {}])\n'.format(lhs, name, body, for_loops)

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
        for_loops = ' '.join(['for {} in {}'.format(self._print(idx), self._print(iters))
                        for idx, iters in zip(expr.indices, iterators)])

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

        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        if isinstance(rhs, Variable) and rhs.rank>1 and rhs.order != lhs.order:
            return'{0} = {1}.T\n'.format(lhs_code,rhs_code)
        else:
            return'{0} = {1}\n'.format(lhs_code,rhs_code)

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
        return ''

    def _print_Deallocate(self, expr):
        return ''

    def _print_NumpyArray(self, expr):
        name = self._get_numpy_name(expr)

        arg   = self._print(expr.arg)
        dtype = self._print_dtype_argument(expr, expr.init_dtype)
        order = f"order='{expr.order}'" if expr.order else ''
        args  = ', '.join(a for a in [arg, dtype, order] if a!= '')
        return f"{name}({args})"

    def _print_NumpyAutoFill(self, expr):
        func_name = self._aliases.get(type(expr), expr.name)

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

    def _print_PyccelInternalFunction(self, expr):
        name = self._aliases.get(type(expr),expr.name)
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_NumpyResultType(self, expr):
        args = expr.args
        if len(args) == 1:
            arg = args[0]
            arg_code = self._print(arg)
            if isinstance(arg, Variable):
                return f"{arg_code}.dtype"
            else:
                return f"({arg_code}).dtype"
        else:
            name = self._aliases.get(type(expr),expr.name)
            args = ', '.join(self._print(a) for a in expr.args)
            return "{}({})".format(name, args)

    def _print_NumpyRandint(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        if expr.low:
            args = "{}, ".format(self._print(expr.low))
        else:
            args = ""
        args += "{}".format(self._print(expr.high))
        if expr.rank != 0:
            size = self._print(expr.shape)
            args += ", size = {}".format(size)
        return "{}({})".format(name, args)

    def _print_NumpyNorm(self, expr):
        name = self._aliases.get(type(expr), expr.name)
        axis = self._print(expr.axis) if expr.axis else None
        if axis:
            return  "{name}({arg},axis={axis})".format(name = name, arg  = self._print(expr.python_arg), axis=axis)
        return  "{name}({arg})".format(name = name, arg  = self._print(expr.python_arg))

    def _print_NumpyNonZero(self, expr):
        name = self._aliases.get(type(expr),'nonzero')
        if name == 'nonzero':
            self.insert_new_import(
                    source = 'numpy',
                    target = AsName(NumpyNonZero, 'nonzero'))
        arg = self._print(expr.array)
        return "{}({})".format(name, arg)

    def _print_NumpyCountNonZero(self, expr):
        name = self._aliases.get(type(expr),'count_nonzero')
        if name == 'count_nonzero':
            self.insert_new_import(
                    source = 'numpy',
                    target = AsName(NumpyNonZero, 'count_nonzero'))

        axis_arg = expr.axis

        arr = self._print(expr.array)
        axis = '' if axis_arg is None else (self._print(axis_arg) + ', ')
        keep_dims = 'keepdims = {}'.format(self._print(expr.keep_dims))

        arg = '{}, {}{}'.format(arr, axis, keep_dims)

        return "{}({})".format(name, arg)

    def _print_Slice(self, expr):
        start = self._print(expr.start) if expr.start else ''
        stop  = self._print(expr.stop)  if expr.stop  else ''
        step  = self._print(expr.step)  if expr.step  else ''
        return '{start}:{stop}:{step}'.format(
                start = start,
                stop  = stop,
                step  = step)

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
        precision = expr.precision

        if not isinstance(expr, (LiteralInteger, LiteralFloat, LiteralComplex)) or \
                precision == -1:
            return repr(expr.python_value)
        else:
            cast_func = DtypePrecisionToCastFunction[dtype.name][precision]
            type_name = cast_func.__name__.lower()
            is_numpy  = type_name.startswith('numpy')
            cast_name = cast_func.name
            name = self._aliases.get(cast_func, cast_name)
            if is_numpy and name == cast_name:
                self.insert_new_import(
                        source = 'numpy', 
                        target = AsName(cast_func, cast_name))
            return '{}({})'.format(name, repr(expr.python_value))

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
        # Print interface functions (one function with multiple decorators describes the problem)
        imports  = ''.join(self._print(i) for i in expr.imports)
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

        imports += ''.join(self._print(i) for i in self.get_additional_imports())

        body = ''.join((interfaces, funcs, classes, init_body))

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

    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_PythonType(self, expr):
        return 'type({})'.format(self._print(expr.arg))
    
    #-----------------Class Printer---------------------------------

    def _print_ClassDef(self, expr):
        classDefName = 'class {}({}):'.format(expr.name,', '.join(self._print(arg) for arg in  expr.superclasses))
        methods = ''.join(self._print(method) for method in expr.methods)
        methods = self._indent_codestring(methods)
        interfaces = ''.join(self._print(method) for method in expr.interfaces)
        interfaces = self._indent_codestring(interfaces)
        classDef = '\n'.join([classDefName, methods, interfaces]) + '\n'
        return classDef

    def _print_ConstructorCall(self, expr):
        cls_name = expr.func.cls_name
        args = ', '.join(self._print(arg) for arg in expr.arguments)
        return f"{cls_name}({args})"

    def _print_Del(self, expr):
        return ''.join(f'del {var}\n' for var in expr.variables)

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

#==============================================================================
def pycode(expr, assign_to=None, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    return PythonCodePrinter(settings).doprint(expr, assign_to)
