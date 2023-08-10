#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
import os

from pyccel.codegen.printing.fcode  import FCodePrinter
from pyccel.codegen.printing.ccode  import CCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter

from pyccel.ast.core      import FunctionDef, Interface, ModuleHeader
from pyccel.errors.errors import Errors
from pyccel.utilities.stage import PyccelStage

_extension_registry = {'fortran': 'f90', 'c':'c',  'python':'py'}
_header_extension_registry = {'fortran': None, 'c':'h',  'python':None}
printer_registry    = {
                        'fortran':FCodePrinter,
                        'c':CCodePrinter,
                        'python':PythonCodePrinter
                      }

pyccel_stage = PyccelStage()

class Codegen(object):

    """Abstract class for code generator."""

    def __init__(self, parser, name):
        """Constructor for Codegen.

        parser: pyccel parser


        name: str
            name of the generated module or program.
        """
        pyccel_stage.set_stage('codegen')
        self._parser   = parser
        self._ast      = parser.ast
        self._name     = name
        self._printer  = None
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
        self._is_program = self.ast.program is not None


    @property
    def parser(self):
        return self._parser

    @property
    def name(self):
        """Returns the name associated to the source code"""

        return self._name

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
    def is_program(self):
        """Returns True if a Program."""

        return self._is_program

    @property
    def ast(self):
        """Returns the AST."""

        return self._ast

    @property
    def language(self):
        """Returns the used language"""

        return self._language

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
        self._printer = code_printer(self.parser.filename, **settings)

    def get_printer_imports(self):
        """return the imports of the current codeprinter"""
        return self._printer.get_additional_imports()

    def _collect_statements(self):
        """Collects statements and split them into routines, classes, etc."""

        scope  = self.parser.scope

        funcs      = []
        interfaces = []


        for i in scope.functions.values():
            if isinstance(i, FunctionDef) and not i.is_header:
                funcs.append(i)
            elif isinstance(i, Interface):
                interfaces.append(i)

        self._stmts['imports'   ] = list(scope.imports['imports'].values())
        self._stmts['variables' ] = list(self.parser.get_variables(scope))
        self._stmts['routines'  ] = funcs
        self._stmts['classes'   ] = list(scope.classes.values())
        self._stmts['interfaces'] = interfaces
        self._stmts['body']       = self.ast

    def doprint(self, **settings):
        """Prints the code in the target language."""
        if not self._printer:
            self.set_printer(**settings)
        return self._printer.doprint(self.ast)


    def export(self, filename=None, **settings):
        """Export code in filename"""
        self.set_printer(**settings)
        ext = _extension_registry[self._language]
        header_ext = _header_extension_registry[self._language]

        if filename is None: filename = self.name
        header_filename = '{name}.{ext}'.format(name=filename, ext=header_ext)
        filename = '{name}.{ext}'.format(name=filename, ext=ext)

        # print module header
        if header_ext is not None:
            code = self._printer.doprint(ModuleHeader(self.ast))
            with open(header_filename, 'w') as f:
                for line in code:
                    f.write(line)

        # print module
        code = self._printer.doprint(self.ast)
        with open(filename, 'w') as f:
            for line in code:
                f.write(line)
                print(line)

        # print program
        prog_filename = None
        if self.is_program and self.language != 'python':
            folder = os.path.dirname(filename)
            fname  = os.path.basename(filename)
            prog_filename = os.path.join(folder,"prog_"+fname)
            code = self._printer.doprint(self.ast.program)
            with open(prog_filename, 'w') as f:
                for line in code:
                    f.write(line)

        return filename, prog_filename
