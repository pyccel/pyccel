# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
from pyccel.ast.core     import Assign, Declare
from pyccel.ast.literals import Nil
from pyccel.ast.low_level_tools import UnpackManagedMemory
from pyccel.ast.utilities import expand_to_loops
from pyccel.ast.variable import Variable
from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors

errors = Errors()

class CppCodePrinter(CodePrinter):
    """
    A printer for printing code in C.

    A printer to convert Pyccel's AST to strings of c code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    verbose : int
        The level of verbosity.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename, *, verbose):

        errors.set_target(filename)

        super().__init__(verbose)

        self._additional_imports = {}
        self._additional_code = ''
        self._in_header = False
        self._declared_vars = []

    def set_scope(self, scope):
        self._declared_vars.append(set())
        super().set_scope(scope)

    def exit_scope(self):
        super().exit_scope()
        self._declared_vars.pop()

    def _indent_codestring(self, lines):
        tab = ' '*self._default_settings['tabwidth']
        if lines == '':
            return lines
        else:
            # lines ends with \n
            return tab+lines.replace('\n','\n'+tab).rstrip(' ')

    def _format_code(self, code):
        return code

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the C++ representation of the function signature.

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
        name = expr.name
        result_var = expr.results.var

        args = ', '.join(self._print(a) for a in expr.arguments)

        result = 'void' if result_var is Nil() else self._print(result_var.class_type)

        return f'{result} {name}({args})'

    def get_declare_type(self, var):
        class_type = self._print(var.class_type)
        const = ' const' if var.is_const else ''

        return f'{class_type}{const}'

    #-----------------------------------------------------------------------
    #                              Print methods
    #-----------------------------------------------------------------------

    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True

        decls = [Declare(v, external=True, module_variable=True) for v in expr.module.variables if not v.is_private]
        global_variables = ''.join(self._print(d) for d in decls)

        classes = '\n'.join(self._print(classDef) for classDef in expr.module.classes)

        funcs = '\n'.join(f"{self.function_signature(f)};" for f in expr.module.funcs if not f.is_inline)

        imports = ''.join(self._print(i) for i in expr.module.imports)

        self.exit_scope()
        self._in_header = False

        sections = ('#pragma once\n',
                    imports,
                    global_variables,
                    classes,
                    funcs)

        return '\n'.join(s for s in sections if s)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        name = expr.name

        global_variables = ''.join([self._print(d) for d in expr.declarations])
        body    = ''.join(self._print(i) for i in expr.body)

        self.exit_scope()

        return ''.join((f'namespace {name} {{\n\n',
                        global_variables,
                        body,
                        '\n}\n'))

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''

        self.set_scope(expr.scope)

        body  = self._print(expr.body)

        self.exit_scope()

        return ''.join((self.function_signature(expr),
                        ' {\n',
                        self._indent_codestring(body),
                        '}\n'))

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs = expand_to_loops(expr,
                    self.scope.get_temporary_variable, self.scope,
                    language_has_vectors = False)
        else:
            body_exprs = expr.body
        body_code = ''
        for b in body_exprs :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body_code += code
        return body_code

    def _print_Pass(self, expr):
        return '// pass\n'

    def _print_Return(self, expr):
        if expr.stmt:
            to_print = [l for l in expr.stmt.body if not ((isinstance(l, Assign) and isinstance(l.lhs, Variable))
                                                        or isinstance(l, UnpackManagedMemory))]
            assigns = {a.lhs: a.rhs for a in expr.stmt.body if (isinstance(a, Assign) and isinstance(a.lhs, Variable))}
            assigns.update({a.out_ptr: a.managed_object for a in expr.stmt.body if isinstance(a, UnpackManagedMemory)})
            prelude = ''.join(self._print(l) for l in to_print)
        else:
            assigns = {}
            prelude = ''

        if expr.expr is None:
            return 'return;\n'

        return_code = self._print(assigns.get(expr.expr, expr.expr))

        return prelude + f'return {return_code};\n'

    def _print_Assign(self, expr):
        lhs = expr.lhs

        prefix = ''
        if lhs in self.scope.variables.values() and lhs not in self._declared_vars[-1]:
            prefix = self.get_declare_type(lhs) + ' '
            self._declared_vars[-1].add(lhs)

        lhs_code = self._print(lhs)
        rhs_code = self._print(expr.rhs)
        return f'{prefix}{lhs_code} = {rhs_code};\n'

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        return ' - '.join(self._print(a) for a in expr.args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_Variable(self, expr):
        name = expr.name
        if expr.is_alias:
            return f'(*{name})'
        else:
            return name

    def _print_Declare(self, expr):
        var = expr.variable

        name = var.name
        class_type = self._print(var.class_type)
        const = ' const' if var.is_const else ''

        return f'{class_type}{const} {name};\n'

    def _print_Literal(self, expr):
        #TODO: Ensure correct precision
        return repr(expr.python_value)

    def _print_PythonNativeBool(self, expr):
        return 'bool'

    def _print_PythonNativeInt(self, expr):
        #TODO: Improve, wrong precision
        return 'int'

    def _print_PythonNativeFloat(self, expr):
        return 'float'

    def _print_PythonNativeComplex(self, expr):
        return 'complex'

    def _print_StringType(self, expr):
        return 'str'
