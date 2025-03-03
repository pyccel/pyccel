from pyccel.ast.core import EmptyNode
from pyccel.errors.errors import Errors
from pyccel.utilities.extensions import Extension
from . import openmp_4_5
from . import openmp_5_0
errors = Errors()


class Openmp(Extension):
    _default_version = 4.5

    def __init__(self):
        self._options = {}

    def set_options(self, **options):
        self._options = options

    def extend_syntax_parser(self, sp):
        if 'openmp' not in self._options.get('accelerators', []):
            return sp

        omp_version = self._options.get('omp_version', self._default_version)
        mixin = openmp_5_0.SyntaxParser if omp_version == 5.0 else openmp_4_5.SyntaxParser


        class Extended(sp, mixin):
            def __init__(self, *args, **kwargs):
                mixin.__init__(self)
                sp.__init__(self, *args, **kwargs)

            def _visit(self, stmt):
                if self._skip_stmts_count:
                    self._skip_stmts_count -= 1
                    return EmptyNode()
                return(sp._visit(self, stmt))

            def _visit_CommentLine(self, stmt):
                line = stmt.s
                if line.startswith('#$'):
                    line = line[2:].lstrip()
                    if line.startswith('omp'):
                        return mixin._visit_CommentLine(self, stmt)
                return sp._visit_CommentLine(self, stmt)
        return Extended

    def extend_semantic_parser(self, sp):
        if 'openmp' not in self._options.get('accelerators', []):
                return sp

        omp_version = self._options.get('omp_version', self._default_version)
        mixin = openmp_5_0.SemanticParser if omp_version == 5.0 else openmp_4_5.SemanticParser

        class Extended(sp, mixin):
            def __init__(self, *args, **kwargs):
                mixin.__init__(self)
                sp.__init__(self, *args, **kwargs)
        return Extended

    def extend_printer(self, printer):
        if 'openmp' not in self._options.get('accelerators', []):
                return printer

        #don't extend
        omp_version = self._options.get('omp_version', self._default_version)
        openmp = openmp_5_0 if omp_version == 5.0 else openmp_4_5

        if self._options['language'] == 'c':
            mixin = openmp.CCodePrinter
        elif self._options['language'] == 'python':
            mixin = openmp.PythonCodePrinter
        elif self._options['language'] == 'fortran' or self._options['language'] is None:
            mixin = openmp.FCodePrinter
        else:
            return printer

        class Extended(printer, mixin):
            def __init__(self, *args, **kwargs):
                mixin.__init__(self)
                printer.__init__(self, *args, **kwargs)
        return Extended
