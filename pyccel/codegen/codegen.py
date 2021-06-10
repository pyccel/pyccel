#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.codegen.printing.fcode  import FCodePrinter
from pyccel.codegen.printing.ccode  import CCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter

from pyccel.ast.basic     import PyccelAstNode
from pyccel.ast.core      import FunctionDef, Module, Program, Interface, ModuleHeader
from pyccel.ast.core      import EmptyNode, Comment, CommentBlock
from pyccel.ast.headers   import Header
from pyccel.errors.errors import Errors

_extension_registry = {'fortran': 'f90', 'c':'c',  'python':'py'}
_header_extension_registry = {'fortran': None, 'c':'h',  'python':None}
printer_registry    = {
                        'fortran':FCodePrinter,
                        'c':CCodePrinter,
                        'python':PythonCodePrinter
                      }


class Codegen(object):

    """Abstract class for code generator."""

    def __init__(self, parser, name):
        """Constructor for Codegen.

        parser: pyccel parser


        name: str
            name of the generated module or program.
        """
        PyccelAstNode.stage = 'codegen'
        self._parser   = parser
        self._ast      = parser.ast
        self._name     = name
        self._printer  = None
        self._kind     = None
        self._code     = None
        self._language = None

        #TODO verify module name != function name
        #it generates a compilation error

        self._stmts = {}
        _structs = [
            'imports',
            'body',
            'routines',
            'classes',
            'modules',
            'variables',
            'interfaces',
            ]
        for key in _structs:
            self._stmts[key] = []

        self._collect_statements()
        self._set_kind()


    @property
    def parser(self):
        return self._parser

    @property
    def name(self):
        """Returns the name associated to the source code"""

        return self._name

    @property
    def kind(self):
        """Returns the kind of the source code: Module, Program or None."""

        return self._kind

    @property
    def imports(self):
        """Returns the imports of the source code."""

        return self._stmts['imports']

    @property
    def variables(self):
        """Returns the variables of the source code."""

        return self._stmts['variables']

    @property
    def body(self):
        """Returns the body of the source code, if it is a Program or Module."""

        return self._stmts['body']

    @property
    def routines(self):
        """Returns functions/subroutines."""

        return self._stmts['routines']

    @property
    def classes(self):
        """Returns the classes if Module."""

        return self._stmts['classes']

    @property
    def interfaces(self):
        """Returns the interfaces."""

        return self._stmts['interfaces']

    @property
    def modules(self):
        """Returns the modules if Program."""

        return self._stmts['modules']

    @property
    def is_module(self):
        """Returns True if a Module."""

        return self.kind == 'module'

    @property
    def is_program(self):
        """Returns True if a Program."""

        return self.kind == 'program'

    @property
    def ast(self):
        """Returns the AST."""

        return self._ast

    @property
    def expr(self):
        """Returns the AST after Module/Program treatment."""

        return self._expr

    @property
    def language(self):
        """Returns the used language"""

        return self._language

    @property
    def code(self):
        """Returns the generated code."""

        return self._code

    def set_printer(self, **settings):
        """ Set the current codeprinter instance"""
        # Get language used (default language used is fortran)
        language = settings.pop('language', 'fortran')

        # Set language
        if not language in ['fortran', 'c', 'python']:
            raise ValueError('{} language is not available'.format(language))
        self._language = language

        # instantiate codePrinter
        code_printer = printer_registry[language]
        errors = Errors()
        errors.set_parser_stage('codegen')
        # set the code printer
        self._printer = code_printer(self.parser, **settings)

    def get_printer_imports(self):
        """return the imports of the current codeprinter"""
        return self._printer.get_additional_imports()

    def _collect_statements(self):
        """Collects statements and split them into routines, classes, etc."""

        namespace  = self.parser.namespace

        funcs      = []
        interfaces = []


        for i in namespace.functions.values():
            if isinstance(i, FunctionDef) and not i.is_header:
                funcs.append(i)
            elif isinstance(i, Interface):
                interfaces.append(i)

        self._stmts['imports'   ] = list(namespace.imports['imports'].values())
        self._stmts['variables' ] = list(self.parser.get_variables(namespace))
        self._stmts['routines'  ] = funcs
        self._stmts['classes'   ] = list(namespace.classes.values())
        self._stmts['interfaces'] = interfaces
        self._stmts['body']       = self.ast


    def _set_kind(self):
        """Finds the source code kind."""


        cls = (Header, EmptyNode, Comment, CommentBlock, Module)
        is_module = all(isinstance(i,cls) for i in self.ast.body)



        if is_module:
            self._kind = 'module'
        else:
            self._kind = 'program'

        #  ...

        #  ...

        expr = None

        if self.is_module:
            expr = Module(
                self.name,
                self.variables,
                self.routines,
                self.interfaces,
                self.classes,
                imports=self.imports)

        elif self.is_program:
            expr = Program(
                self.name,
                self.variables,
                self.body.body,
                imports=self.imports)

        else:
            raise NotImplementedError('TODO')


        self._expr = expr

        #  ...

    def doprint(self, **settings):
        """Prints the code in the target language."""
        if not self._printer:
            self.set_printer(**settings)
        self._code = self._printer.doprint(self.expr)
        return self._code

    def export(self, filename=None, **settings):
        """Export code in filename"""
        self.set_printer(**settings)
        ext = _extension_registry[self._language]
        header_ext = _header_extension_registry[self._language]

        if filename is None: filename = self.name
        header_filename = '{name}.{ext}'.format(name=filename, ext=header_ext)
        filename = '{name}.{ext}'.format(name=filename, ext=ext)

        if header_ext is not None and self.is_module:
            code = self._printer.doprint(ModuleHeader(self.expr))
            with open(header_filename, 'w') as f:
                for line in code:
                    f.write(line)
        # print module or program code
        self._code = self._printer.doprint(self.expr)
        with open(filename, 'w') as f:
            for line in self._code:
                f.write(line)

        return filename
