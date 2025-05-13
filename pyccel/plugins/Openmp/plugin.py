import inspect
import os

from pyccel.errors.errors import Errors
from pyccel.utilities.plugins import Plugin
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.semantic import SemanticParser
from pyccel.codegen.printing.ccode import CCodePrinter
from pyccel.codegen.printing.fcode import FCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter

errors = Errors()

class Openmp(Plugin):
    _default_version = 4.5
    _versions = {
        4.5: {
            'SyntaxParser': openmp_4_5.SyntaxParser,
            'SemanticParser': openmp_4_5.SemanticParser,
            'CCodePrinter': openmp_4_5.CCodePrinter,
            'FCodePrinter': openmp_4_5.FCodePrinter,
            'PythonCodePrinter': openmp_4_5.PythonCodePrinter
        },
        5.0: {
            'SyntaxParser': openmp_5_0.SyntaxParser,
            'SemanticParser': openmp_5_0.SemanticParser,
            'CCodePrinter': openmp_5_0.CCodePrinter,
            'FCodePrinter': openmp_5_0.FCodePrinter,
            'PythonCodePrinter': openmp_5_0.PythonCodePrinter
        }
    }

    def __init__(self):
        self._loaded = False
        self._options = {}
        self._pyccel_parsers = [SyntaxParser, SemanticParser, CCodePrinter, FCodePrinter, PythonCodePrinter]

    def handle_loading(self, options):
        self._options.clear()
        self._options.update(options)
        if self._options.get('clear', False):
            omp_version = list(self._versions.values())[0]
            for omp_parser in omp_version:
                for parser in self._pyccel_parsers:
                    if parser.__name__ == omp_parser:
                        ex = omp_version[omp_parser]
                        if hasattr(ex, 'setup'):
                            parser.__init__ = getattr(ex, 'setup')(self._options, parser.__init__)
                        for name, method in inspect.getmembers(ex, predicate=inspect.isfunction):
                            original_method = ex.helper_check_config(method, self._options, None)
                            if original_method:
                                setattr(parser, name, original_method)
                            else:
                                delattr(parser, name)
            self._loaded = False
            return
        if not self._options.get('omp_version', None):
            self._options['omp_version'] = float(os.environ.get('PYCCEL_OMP_VERSION', self._default_version))
        if self._options['omp_version'] not in self._versions:
            errors.report(
                f"OPENMP {self._options['omp_version']} is not supported. defaulting to OPENMP {self._default_version} instead.",
                severity='warning')
            self._options['omp_version'] = self._default_version
        if self._loaded or 'openmp' not in self._options['accelerators']:
            return
        self._loaded = True
        for omp_version in self._versions:
            for omp_parser in self._versions[omp_version]:
                for parser in self._pyccel_parsers:
                    if parser.__name__ == omp_parser:
                        ex = self._versions[omp_version][omp_parser]
                        if hasattr(ex, 'setup'):
                            parser.__init__ = getattr(ex, 'setup')(self._options, parser.__init__)
                        for name, method in inspect.getmembers(ex, predicate=inspect.isfunction):
                            original_method = getattr(parser, name, None)
                            decorated_method = ex.helper_check_config(method, self._options, original_method)
                            setattr(parser, name, decorated_method)

    @property
    def loaded(self):
        return self._loaded
