from .openmp_4_5 import SyntaxParser as PSyntaxParser
from .openmp_4_5 import SemanticParser as PSemanticParser
from .openmp_4_5 import CCodePrinter as PCCodePrinter
from .openmp_4_5 import FCodePrinter as PFCodePrinter
from .openmp_4_5 import PythonCodePrinter as PPythonCodePrinter
from .omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList
from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx import metamodel_for_language

class SyntaxParser(PSyntaxParser):
    """Openmp 5.0 syntax parser"""
    def __init__(self):
        this_folder = dirname(__file__)
        # Get metamodel from language description
        grammar = join(this_folder, "grammar/openmp.tx")
        omp_classes = [OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList]
        self._omp_metamodel = metamodel_from_file(grammar, classes=omp_classes)

        # object processors: are registered for particular classes (grammar rules)
        # and are called when the objects of the given class is instantiated.
        # The rules OMP_X_Y are used to insert the version of the syntax used

        textx_mm = metamodel_for_language('textx')
        grammar_model = textx_mm.grammar_model_from_file(grammar)

        obj_processors = {r.name: (lambda r: lambda _: r.name.replace('_PARENT', '').lower())(r)
                          for r in grammar_model.rules if r.name.endswith('_PARENT')}

        obj_processors.update({
            'OMP_4_5': lambda _: 4.5,
            'OMP_5_0': lambda _: 5.0,
            'OMP_5_1': lambda _: 5.1,
            'TRUE': lambda _: True,
            'OMP_VERSION': lambda _: 5.0,
        })
        self._omp_metamodel.register_obj_processors(obj_processors)
        self._pending_directives = []
        self._skip_stmts_count = 0


class SemanticParser(PSemanticParser):
    """Openmp 5.0 semantic parser"""
    pass


class CCodePrinter(PCCodePrinter):
    """Openmp 5.0 C printer"""

    pass

class FCodePrinter(PFCodePrinter):
    """Openmp 5.0 fortran printer"""
    pass

class PythonCodePrinter(PPythonCodePrinter):
    """Openmp 5.0 python printer"""
    pass
