"""Classes that handles loading the Openmp Plugin"""
import inspect
import os
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, List

from pyccel.errors.errors import Errors
from pyccel.errors.messages import OMP_VERSION_NOT_SUPPORTED
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.utilities.plugins import Plugin, PatchRegistry, PatchInfo

__all__ = (
    "OmpPatchInfo",
    "OmpPatchRegistry",
    "Openmp",
)

errors = Errors()

@dataclass
class OmpPatchInfo(PatchInfo):
    """Store information about a single patch"""
    version: float


@dataclass
class OmpPatchRegistry(PatchRegistry):
    """Registry for all omp patches applied to a single class"""
    patches: Dict[str, List[OmpPatchInfo]] = field(default_factory=dict)
    loaded_versions: List[float] = field(default_factory=list)
    """Registry for all patches applied to a single class"""

    def get_patches_for_version(self, version):
        """Get all patches for a specific version"""
        version_patches = {method_name: [patch for patch in patch_list if patch.version == version] for
                           method_name, patch_list in self.patches.items()}
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
        self.loaded_versions.remove(version)


class Openmp(Plugin):
    """
    Provides functionality for integrating OpenMP-specific features into parsers within Pyccel.

    Attributes
    ----------
    DEFAULT_VERSION : float
        The default OpenMP version to use if no specific version is requested.
    VERSION_MODULES : dict
        A mapping of OpenMP versions to their corresponding implementation modules.
    _options : dict
        Configuration options passed to the OpenMP plugin.
    """
    __slots__ = ("_options",)
    DEFAULT_VERSION = 4.5

    VERSION_MODULES = {
        4.5: openmp_4_5,
        5.0: openmp_5_0
    }

    def __init__(self):
        self._options = {}
        self._patch_registries = []
        self._refresh = False
        super().__init__()

    def set_options(self, options):
        """set options for the plugin, this may impact its targets"""
        assert isinstance(options, dict)
        self._options.clear()
        self._options.update(options)

    def register(self, instances):
        """Register the provided instances with the OpenMP plugin"""
        new_patch_regs = [OmpPatchRegistry(instance) for instance in instances
                            if not any(reg.target is instance for reg in self._patch_registries)]
        self._patch_registries += new_patch_regs
        if self._refresh:
            new_patch_regs = self._patch_registries
        version = self._resolve_version()
        if 'openmp' not in self._options.get('accelerators', []):
            return
        for reg in new_patch_regs:
            if version in reg.loaded_versions:
                continue
            self._apply_patches(version, reg)
            reg.loaded_versions.append(version)

    def refresh(self):
        """refresh all registered targets with current options"""
        self._refresh = True
        self.register(self.get_all_targets())
        self._refresh = False

    def unregister(self, instances):
        """Unregister the provided instances"""
        targets = [reg for reg in self._patch_registries if reg.target in instances]
        for reg in targets:
            for ver in reg.loaded_versions:
                self._unload_patches(reg, ver)
        self._patch_registries = [reg for reg in self._patch_registries if reg not in targets]

    def _resolve_version(self):
        """Determine which OpenMP version to use based on options or environment"""
        requested_version = self._options.get('omp_version', None)

        if not requested_version:
            env_version = os.environ.get('PYCCEL_OMP_VERSION', None)
            requested_version = float(env_version) if env_version else self.DEFAULT_VERSION
            self._options['omp_version'] = requested_version

        if requested_version not in self.VERSION_MODULES:
            errors.report(
                OMP_VERSION_NOT_SUPPORTED.format(requested_version, self.DEFAULT_VERSION),
                severity='warning')
            self._options['omp_version'] = self.DEFAULT_VERSION
            return self.DEFAULT_VERSION

        return requested_version

    def _apply_patches(self, version, registry):
        """Apply patches from the specified version module to parser classes"""
        module = self.VERSION_MODULES[version]
        parser = registry.target
        parser_name = parser.__class__.__name__
        impl = getattr(module, parser_name, None)

        if not impl:
            return
        if hasattr(impl, 'setup'):
            impl.setup(self._options, parser)

        for name, method in inspect.getmembers(impl, predicate=inspect.isfunction):
            original_method = getattr(parser, name, None)
            decorated_method = impl.helper_check_config(method, self._options, original_method)
            setattr(parser, name, MethodType(decorated_method, parser))

            patch_info = OmpPatchInfo(
                original_method=original_method,
                patched_method=decorated_method,
                version=version,
                method_name=name
            )
            registry.register_patch(name, patch_info)

    def _unload_patches(self, registry, version):
        """Unload patches for a specific version"""
        parser = registry.target
        version_patches = registry.get_patches_for_version(version)
        for method_name, info in version_patches.items():
            for patch_info in info:
                if patch_info.is_new_method:
                    if hasattr(parser, method_name):
                        delattr(parser, method_name)
                else:
                    original = patch_info.original_method
                    if original:
                        setattr(parser, method_name, original)
        registry.unregister_patches_for_version(version)
