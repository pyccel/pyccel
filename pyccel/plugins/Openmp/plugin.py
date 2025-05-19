"""Classes that handles loading the Openmp Plugin"""
import inspect
import os

from pyccel.codegen.printing.ccode import CCodePrinter
from pyccel.codegen.printing.fcode import FCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter
from pyccel.errors.errors import Errors
from pyccel.parser.semantic import SemanticParser
from pyccel.parser.syntactic import SyntaxParser
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.utilities.plugins import Plugin

errors = Errors()


class Openmp(Plugin):
    """
    Provides functionality for integrating OpenMP-specific features into parsers within Pyccel.

    Attributes
    ----------
    DEFAULT_VERSION : float
        The default OpenMP version to use if no specific version is requested.
    VERSION_MODULES : dict
        A mapping of OpenMP versions to their corresponding implementation modules.
    PARSER_TYPES : list
        A list of parser classes that the OpenMP plugin supports.
    _options : dict
        Configuration options passed to the OpenMP plugin.
    _loaded_versions : list
        A list containing the OpenMP versions currently loaded.
    """
    __slots__ = ("_options", "_loaded_versions")
    DEFAULT_VERSION = 4.5

    VERSION_MODULES = {
        4.5: openmp_4_5,
        5.0: openmp_5_0
    }

    PARSER_TYPES = [SyntaxParser, SemanticParser, CCodePrinter, FCodePrinter, PythonCodePrinter]

    def __init__(self):
        self._options = {}
        self._loaded_versions = []

    def handle_loading(self, options):
        """Handle the loading and unloading openmp versions."""
        self._options.clear()
        self._options.update(options)
        if self._options.get('clear', False):
            self._unload_patches()
            return

        version = self._resolve_version()
        if 'openmp' not in self._options.get('accelerators', []) or version in self._loaded_versions:
            return
        self._loaded_versions.append(version)
        self._apply_patches(version)

    def _resolve_version(self):
        """Determine which OpenMP version to use based on options or environment"""
        requested_version = self._options.get('omp_version', None)

        if not requested_version:
            env_version = os.environ.get('PYCCEL_OMP_VERSION', None)
            requested_version = float(env_version) if env_version else self.DEFAULT_VERSION
            self._options['omp_version'] = requested_version

        if requested_version not in self.VERSION_MODULES:
            errors.report(
                f"OPENMP {requested_version} is not supported. Defaulting to OPENMP {self.DEFAULT_VERSION}.",
                severity='warning')
            self._options['omp_version'] = self.DEFAULT_VERSION
            return self.DEFAULT_VERSION

        return requested_version

    def _apply_patches(self, version):
        """Apply patches from the specified version module to parser classes"""
        module = self.VERSION_MODULES[version]

        for parser_cls in self.PARSER_TYPES:
            parser_name = parser_cls.__name__

            impl = getattr(module, parser_name, None)
            if not impl:
                continue

            if hasattr(impl, 'setup'):
                parser_cls.__init__ = getattr(impl, 'setup')(self._options, parser_cls.__init__)

            for name, method in inspect.getmembers(impl, predicate=inspect.isfunction):
                original_method = getattr(parser_cls, name, None)
                decorated_method = impl.helper_check_config(method, self._options, original_method)
                setattr(parser_cls, name, decorated_method)

    def _unload_patches(self):
        """Remove patches applied to parser classes"""
        if not self._loaded_versions:
            return
        version = self._loaded_versions[0]
        module = self.VERSION_MODULES[version]

        for parser_cls in self.PARSER_TYPES:
            parser_name = parser_cls.__name__
            impl = getattr(module, parser_name, None)
            if not impl:
                continue
            if hasattr(impl, 'setup'):
                parser_cls.__init__ = getattr(impl, 'setup')(self._options, parser_cls.__init__)
            # remove/restore all methods from the implementation
            for name, method in inspect.getmembers(impl, predicate=inspect.isfunction):
                original_method = impl.helper_check_config(method, self._options, None)
                if original_method:
                    setattr(parser_cls, name, original_method)
                else:
                    delattr(parser_cls, name)
        self._loaded_versions = []

    @property
    def loaded_versions(self):
        """Return loaded openmp versions"""
        return self._loaded_versions
