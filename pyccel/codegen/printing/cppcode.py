# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
from pyccel.ast.literals import Nil
from pyccel.ast.utilities import expand_to_loops
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

        result = 'void' if result_var is Nil() else self._print(result_var)

        return f'{result} {name}({args})'

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
