from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx import metamodel_for_language
import functools

from .openmp_4_5 import SyntaxParser as PSyntaxParser
from .openmp_4_5 import SemanticParser as PSemanticParser
from .openmp_4_5 import CCodePrinter as PCCodePrinter
from .openmp_4_5 import FCodePrinter as PFCodePrinter
from .openmp_4_5 import PythonCodePrinter as PPythonCodePrinter
from .omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList

class SyntaxParser(PSyntaxParser):
    """Openmp 5.0 syntax parser"""
    _version = 5.0
    _method_registry = {}
    @classmethod
    def setup(cls, options, method=None, clear=False):
        if options.get('clear', False):
            return cls._method_registry[method.__name__]
        cls._method_registry[method.__name__] = method
        this_folder = dirname(__file__)
        # Get metamodel from language description
        grammar = join(this_folder, "grammar/openmp.tx")
        omp_classes = [OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger, OmpList]
        cls._omp_metamodel = metamodel_from_file(grammar, classes=omp_classes)

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
        cls._omp_metamodel.register_obj_processors(obj_processors)
        @functools.wraps(method)
        def setup(instance, *args, **kwargs):
            if cls._version != options.get('omp_version', None) or not 'openmp' in options.get('accelerators'):
                method(instance, *args, **kwargs)
                return
            instance._skip_stmts_count = 0
            method(instance, *args, **kwargs)
        return setup

class SemanticParser(PSemanticParser):
    """Openmp 5.0 semantic parser"""
    _version = 5.0
    _method_registry = {}

class CCodePrinter(PCCodePrinter):
    """Openmp 5.0 C printer"""
    _version = 5.0
    _method_registry = {}

class FCodePrinter(PFCodePrinter):
    """Openmp 5.0 fortran printer"""
    _version = 5.0
    _method_registry = {}

class PythonCodePrinter(PPythonCodePrinter):
    """Openmp 5.0 python printer"""
    _version = 5.0
    _method_registry = {}
