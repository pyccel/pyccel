#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import os

from pyccel.codegen.printing.fcode  import FCodePrinter
from pyccel.codegen.printing.ccode  import CCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter

from pyccel.ast.core      import FunctionDef, Interface, ModuleHeader
from pyccel.utilities.stage import PyccelStage

_extension_registry = {'fortran': 'f90', 'c':'c',  'python':'py'}
_header_extension_registry = {'fortran': None, 'c':'h',  'python':None}
printer_registry    = {
                        'fortran':FCodePrinter,
                        'c':CCodePrinter,
                        'python':PythonCodePrinter
                      }

pyccel_stage = PyccelStage()

class Codegen:
    """
    Class which handles the generation of code.

    The class which handles the generation of code. This is done by creating an appropriate
    class inheriting from `CodePrinter` and using it to create strings describing the code
    that should be printed. This class then takes care of creating the necessary files.

    Parameters
    ----------
    parser : SemanticParser
        The Pyccel semantic parser for a Python program or module. This contains the
        annotated AST and additional information about the variables scope.
    name : str
        Name of the generated module or program.
    language : str
        The language which the printer should print to.
    """
    def __init__(self, parser, name, language):
        pyccel_stage.set_stage('codegen')
        self._parser   = parser
        self._ast      = parser.ast
        self._name     = name
        self._printer  = None
        self._language = language

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

        # instantiate code_printer
        try:
            CodePrinterSubclass = printer_registry[language]
        except KeyError as err:
            raise ValueError(f'{language} language is not available') from err

        self._printer = CodePrinterSubclass(self.parser.filename)

    @property
    def parser(self):
        """
        The parser which generated the AST printed by this class.

        The parser which generated the AST printed by this class.
        """
        return self._parser

    @property
    def printer(self):
        """
        The printer which is used to generate code.

        The printer which is used by this class to generate code in the target language.
        """
        return self._printer

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


    def export(self, filename):
        """
        Export code to a file with the requested name.

        Generate the code in the target language from the AST and print this code
        to file. Between 1 and 3 files are generated depending on the AST and the
        target language. A source file is always generated. In languages with
        header files, a header file is also generated. Finally if the AST includes
        a program and the target language is not Python a program source file is
        also generated. The source and header files are named by appending the
        extension to the requested filename. The program source file is named by
        additionally prepending 'prog_' to the requested filename.

        Parameters
        ----------
        filename : str
            The base (i.e. no extensions) of the filename of the file where the
            code should be printed to.

        Returns
        -------
        filename : str
            The name of the file where the source code was printed.
        prog_filename : str
            The name of the file where the source code for the program was printed.
        """
        ext = _extension_registry[self._language]
        header_ext = _header_extension_registry[self._language]

        header_filename = f'{filename}.{header_ext}'
        filename = f'{filename}.{ext}'

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
