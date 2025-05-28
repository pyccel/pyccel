"""Classes that handles loading the Openmp Plugin"""
import inspect
import os
from dataclasses import dataclass

from pyccel.codegen.printing.ccode import CCodePrinter
from pyccel.codegen.printing.fcode import FCodePrinter
from pyccel.codegen.printing.pycode import PythonCodePrinter
from pyccel.errors.errors import Errors
from pyccel.parser.semantic import SemanticParser
from pyccel.parser.syntactic import SyntaxParser
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.utilities.plugins import Plugin, ClassPatchRegistry, PatchInfo

errors = Errors()

@dataclass
class OmpClassPatchRegistry(ClassPatchRegistry):
    def get_patches_for_version(self, version):
        """Get all patches for a specific version"""
        version_patches = {}
        for method_name, patch_list in self.patches.items():
            for patch in patch_list:
                if patch.version == version:
                    version_patches[method_name] = patch
                    break
        return version_patches

    def unregister_patches_for_version(self, version):
        """Remove all patches for a specific version"""
        for method_name in list(self.patches.keys()):
            self.patches[method_name] = [
                patch for patch in self.patches[method_name]
                if patch.version != version
            ]
            if not self.patches[method_name]:
                del self.patches[method_name]


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

    def __init__(self):
        self._options = {}
        self._loaded_versions = []
        super().__init__()

    def _setup_patch_registries(self):
        """Setup OpenMP-specific patch registries"""
        self._patch_registries = [
            OmpClassPatchRegistry(SyntaxParser),
            OmpClassPatchRegistry(SemanticParser),
            OmpClassPatchRegistry(CCodePrinter),
            OmpClassPatchRegistry(FCodePrinter),
            OmpClassPatchRegistry(PythonCodePrinter)
        ]

    def handle_loading(self, options):
        """Handle the loading and unloading openmp versions."""
        self._options.clear()
        self._options.update(options)
        if self._options.get('clear', False):
            self._unload_all_patches()
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

        for registry in self._patch_registries:
            parser_cls = registry.target_class
            parser_name = parser_cls.__name__
            impl = getattr(module, parser_name, None)
            if not impl:
                continue

            if hasattr(impl, 'setup'):
                original_init = parser_cls.__init__
                patched_init = getattr(impl, 'setup')(self._options, original_init)
                setattr(parser_cls, '__init__', patched_init)

                patch_info = PatchInfo(
                    original_method=original_init,
                    patched_method=patched_init,
                    version=version,
                    method_name='__init__'
                )
                registry.register_patch('__init__', patch_info)

            for name, method in inspect.getmembers(impl, predicate=inspect.isfunction):
                original_method = getattr(parser_cls, name, None)
                decorated_method = impl.helper_check_config(method, self._options, original_method)
                setattr(parser_cls, name, decorated_method)

                patch_info = PatchInfo(
                    original_method=original_method,
                    patched_method=decorated_method,
                    version=version,
                    method_name=name
                )
                registry.register_patch(name, patch_info)

    def _unload_patches_for_version(self, version: float):
        """Unload patches for a specific version"""
        if version not in self._loaded_versions:
            return

        for registry in self._patch_registries:
            parser_cls = registry.target_class
            version_patches = registry.get_patches_for_version(version)

            for method_name, patch_info in version_patches.items():
                if patch_info.is_new_method:
                    if hasattr(parser_cls, method_name):
                        delattr(parser_cls, method_name)
                else:
                    original = patch_info.original_method
                    if original:
                        setattr(parser_cls, method_name, original)

            registry.unregister_patches_for_version(version)

        self._loaded_versions.remove(version)

    def _unload_all_patches(self):
        """Unload all patches and restore original methods"""
        for version in self._loaded_versions:
            self._unload_patches_for_version(version)

    @property
    def loaded_versions(self):
        """Return loaded openmp versions"""
        return self._loaded_versions
