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
    verbose : int
        The level of verbosity.
    """
    def __init__(self, parser, name, language, verbose):
        pyccel_stage.set_stage('codegen')
        self._parser   = parser
        self._ast      = parser.ast
        self._name     = name
        self._language = language
        self._verbose  = verbose
        self._is_program = self.ast.program is not None

        # instantiate code_printer
        try:
            CodePrinterSubclass = printer_registry[language]
        except KeyError as err:
            raise ValueError(f'{language} language is not available') from err

        self._printer = CodePrinterSubclass(self.parser.filename, verbose = self._verbose)

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
    def is_program(self):
        """
        True if the file is a program.

        True if the file is a program, in other words True if the file contains a
        `if __name__ == '__main__'` statement.
        """
        return self._is_program

    @property
    def ast(self):
        """Returns the AST."""

        return self._ast

    def get_printer_imports(self):
        """
        Get the objects that were imported by the current codeprinter.

        Get the objects that were imported by the current codeprinter.
        These imports may affect the necessary compiler commands.

        Returns
        -------
        dict[str, Import]
            A dictionary mapping the include strings to the import module.
        """
        additional_imports = self._printer.get_additional_imports().copy()
        if self._parser.metavars['printer_imports']:
            for i in self._parser.metavars['printer_imports'].split(','):
                additional_imports.setdefault(i.strip(), None)
        return additional_imports


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
        additionally prepending `prog_` to the requested filename.

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
        pyi_filename = f'{filename}.pyi'
        filename = f'{filename}.{ext}'

        if self._verbose:
            print ('>>> Printing :: ', filename)
        # print module
        code = self._printer.doprint(self.ast)
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(code)

        module_header = ModuleHeader(self.ast)
        # print module header
        if header_ext is not None:
            if self._verbose:
                print ('>>> Printing :: ', header_filename)
            code = self._printer.doprint(module_header)
            with open(header_filename, 'w', encoding="utf-8") as f:
                f.write(code)

        if self._verbose:
            print ('>>> Printing :: ', pyi_filename)
        code = printer_registry['python'](self.parser.filename, verbose = self._verbose).doprint(module_header)
        if self._language != 'python':
            printer_imports = ', '.join(self.get_printer_imports().keys())
            if printer_imports:
                code = f'#$ header metavar printer_imports="{printer_imports}"\n' + code
            libdirs = self._parser.metavars.get('libdirs', '')
            if libdirs:
                code = f'#$ header metavar libdirs="{libdirs}"\n' + code
            libs = self._parser.metavars.get('libraries', '')
            if libs:
                code = f'#$ header metavar libraries="{libs}"\n' + code
            incs = self._parser.metavars.get('includes', '')
            if incs:
                code = f'#$ header metavar includes="{incs}"\n' + code
        with open(pyi_filename, 'w', encoding="utf-8") as f:
            f.write(code)

        # print program
        prog_filename = None
        if self.is_program and self._language != 'python':
            folder = os.path.dirname(filename)
            fname  = os.path.basename(filename)
            prog_filename = os.path.join(folder,"prog_"+fname)
            if self._verbose:
                print ('>>> Printing :: ', prog_filename)
            code = self._printer.doprint(self.ast.program)
            with open(prog_filename, 'w') as f:
                f.write(code)

        return filename, prog_filename
