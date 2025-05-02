import inspect
from pyccel.errors.errors import Errors
from pyccel.utilities.extensions import Extension
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.parser.syntactic import SyntaxParser
from pyccel.parser.semantic import SemanticParser
from pyccel.codegen.printing.ccode import CCodePrinter
from pyccel.codegen.printing.fcode import FCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter

errors = Errors()

class Openmp(Extension):
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

    def load(self, options):
        self._options.clear()
        self._options.update(options)
        if self._loaded or 'openmp' not in options['accelerators'] or options['omp_version'] not in self._versions:
            return

        self._loaded = True
        for omp_version in self._versions:
            for omp_parser in self._versions[omp_version]:
                for parser in self._pyccel_parsers:
                    if parser.__name__ == omp_parser:
                        ex = self._versions[omp_version][omp_parser]
                        for name, method in inspect.getmembers(ex, predicate=inspect.ismethod):
                            # skip special and helper methods
                            if name.startswith('__') or name.startswith('_helper'):
                                continue
                            elif name == 'setup':
                                parser.__init__ = method(self._options, parser.__init__)
                                continue
                            original_method = getattr(parser, name, None)
                            decorated_method = method(self._options, original_method)
                            setattr(parser, name, decorated_method)


    @property
    def loaded(self):
        return self._loaded
