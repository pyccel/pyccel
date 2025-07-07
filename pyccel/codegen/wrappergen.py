# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
from pathlib import Path

from .codegen import _extension_registry, _header_extension_registry
from .printing.fcode  import FCodePrinter
from .printing.ccode  import CCodePrinter
from .printing.cwrappercode  import CWrapperCodePrinter
from .wrapper.fortran_to_c_wrapper import FortranToCWrapper
from .wrapper.c_to_python_wrapper import CToPythonWrapper
from ..ast.core import ModuleHeader
from ..naming import name_clash_checkers
from ..parser.scope import Scope
from ..utilities.stage import PyccelStage

wrapper_registry = {
        'fortran' : [FortranToCWrapper, CToPythonWrapper],
        'c' : [CToPythonWrapper],
        'python' : [],
        }

printer_registry = {
        FortranToCWrapper : FCodePrinter,
        CToPythonWrapper : CWrapperCodePrinter,
        }

pyccel_stage = PyccelStage()

class Wrappergen:
    """
    """
    def __init__(self, codegen, name, language, verbose):
        pyccel_stage.set_stage('cwrapper')
        self._ast      = codegen.ast
        self._name     = name
        self._language = language
        self._verbose  = verbose
        self._wrapper_ast = []

        self._wrapper_types = wrapper_registry[language]
        self._printer_types = [printer_registry[w] for w in self._wrapper_types]
        self._additional_imports = [{} for _ in self._wrapper_types]

    def wrap(self, dirpath):
        current_name_clash_checker = Scope.name_clash_checker
        ast = self._ast
        for Wrapper in self._wrapper_types:
            if self._verbose:
                print(f">> Building {Wrapper.start_language}-{Wrapper.target_language} interface :: ", self._name)

            Scope.name_clash_checker = name_clash_checkers[Wrapper.start_language.lower()]
            wrapper = Wrapper(dirpath, verbose = self._verbose)

            ast = wrapper.wrap(ast)
            self._wrapper_ast.append(ast)

        Scope.name_clash_checker = current_name_clash_checker

    def print(self, dirpath):
        dirpath = Path(dirpath)
        files = [dirpath / f'{ast.name}_wrapper.{_extension_registry[Wrapper.start_language.lower()]}'
                 for ast, Wrapper in zip(self._wrapper_ast, self._wrapper_types)]
        for i, (filepath, ast, Printer) in enumerate(zip(files, self._wrapper_ast, self._printer_types)):
            header_ext = _header_extension_registry[Printer.language.lower()]

            if self._verbose:
                print ('>>> Printing :: ', filepath)
            printer = Printer(ast.name, verbose=self._verbose)

            # print module
            code = printer.doprint(ast)

            with open(filepath, 'w', encoding="utf-8") as f:
                f.write(code)

            # print module header
            if header_ext is not None:
                header_filename = f'{ast.name}_wrapper.{header_ext}'
                module_header = ModuleHeader(ast)
                if self._verbose:
                    print ('>>> Printing :: ', header_filename)
                code = printer.doprint(module_header)
                with open(header_filename, 'w', encoding="utf-8") as f:
                    f.write(code)

            self._additional_imports[i] = printer.get_additional_imports().copy()

        return files

    def get_additional_imports(self):
        return self._additional_imports

    @property
    def printed_languages(self):
        return [Printer.language.lower() for Printer in self._printer_types]
