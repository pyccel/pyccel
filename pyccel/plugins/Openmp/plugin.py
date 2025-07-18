"""Classes that handle loading the OpenMP Plugin."""
import inspect
import os
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, List

from pyccel.errors.errors import Errors
from pyccel.errors.messages import OMP_VERSION_NOT_SUPPORTED
from pyccel.plugins.Openmp import openmp_4_5
from pyccel.plugins.Openmp import openmp_5_0
from pyccel.utilities.pluginmanager import Plugin, PatchRegistry, PatchInfo

__all__ = (
    "OmpPatchInfo",
    "OmpPatchRegistry",
    "Openmp",
)

errors = Errors()

@dataclass
class OmpPatchInfo(PatchInfo):
    """
    Store information about a single OpenMP patch.

    This class extends PatchInfo to include version information for OpenMP patches.
    It is used to track which OpenMP version a particular patch is associated with.

    Parameters
    ----------
    original_method : Optional[Callable]
        The original method that is being patched. If None, the patch
        is considered to be for a new method.
    patched_method : Callable
        The new method that will replace the original method.
    method_name : str
        The name of the method being patched.
    version : float
        The OpenMP version associated with this patch.

    See Also
    --------
    PatchInfo : Base class for patch information.
    OmpPatchRegistry : Registry for OpenMP patches.
    """
    version: float


@dataclass
class OmpPatchRegistry(PatchRegistry):
    """
    Registry for all OpenMP patches applied to a single class.

    This class extends PatchRegistry to manage OpenMP-specific patches.
    It tracks which OpenMP versions have been loaded and provides methods
    to manage patches for specific versions.

    Parameters
    ----------
    target : Any
        The target class or object to which patches will be applied.
    patches : Dict[str, List[OmpPatchInfo]], optional
        A dictionary mapping method names to lists of OmpPatchInfo objects.
        Default is an empty dictionary.
    loaded_versions : List[float], optional
        A list of OpenMP versions that have been loaded for this registry.
        Default is an empty list.

    See Also
    --------
    PatchRegistry : Base class for patch registries.
    OmpPatchInfo : Information about a single OpenMP patch.

    Examples
    --------
    >>> class MyClass:
    ...     def method(self):
    ...         return 1
    >>> obj = MyClass()
    >>> registry = OmpPatchRegistry(obj)
    >>> def method(self):
    ...     return 2
    >>> patch_info = OmpPatchInfo(obj.method, method, "method", 4.5)
    >>> registry.register_patch("method", patch_info)
    >>> registry.loaded_versions.append(4.5)
    """
    patches: Dict[str, List[OmpPatchInfo]] = field(default_factory=dict)
    loaded_versions: List[float] = field(default_factory=list)

    def get_patches_for_version(self, version):
        """
        Get all patches for a specific OpenMP version.

        This method filters the patches dictionary to return only those patches
        that are associated with the specified OpenMP version.

        Parameters
        ----------
        version : float
            The OpenMP version to filter patches for.

        Returns
        -------
        Dict[str, List[OmpPatchInfo]]
            A dictionary mapping method names to lists of OmpPatchInfo objects
            for the specified version.

        See Also
        --------
        deregister_patches_for_version : Remove patches for a specific version.

        Examples
        --------
        >>> class MyClass:
        ...     def method(self):
        ...         return 1
        >>> obj = MyClass()
        >>> registry = OmpPatchRegistry(obj)
        >>> def method(self):
        ...     return 2
        >>> patch_info = OmpPatchInfo(obj.method, method, "method", 4.5)
        >>> registry.register_patch("method", patch_info)
        >>> patches = registry.get_patches_for_version(4.5)
        >>> len(patches["method"])
        1
        """
        version_patches = {method_name: [patch for patch in patch_list if patch.version == version] for
                           method_name, patch_list in self.patches.items()}
        return version_patches

    def deregister_patches_for_version(self, version):
        """
        Remove all patches for a specific OpenMP version.

        This method removes all patches associated with the specified OpenMP version
        from the registry and updates the loaded_versions list accordingly.

        Parameters
        ----------
        version : float
            The OpenMP version for which to remove patches.

        See Also
        --------
        get_patches_for_version : Get patches for a specific version.

        Examples
        --------
        >>> class MyClass:
        ...     def method(self):
        ...         return 1
        >>> obj = MyClass()
        >>> registry = OmpPatchRegistry(obj)
        >>> def method(self):
        ...     return 2
        >>> patch_info = OmpPatchInfo(obj.method, method, "method", 4.5)
        >>> registry.register_patch("method", patch_info)
        >>> registry.loaded_versions.append(4.5)
        >>> registry.deregister_patches_for_version(4.5)
        >>> 4.5 in registry.loaded_versions
        False
        >>> "method" in registry.patches
        False
        """
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

    This plugin enables Pyccel to understand and process OpenMP directives in source code.
    It supports different versions of the OpenMP standard and applies appropriate patches
    to parser classes to handle OpenMP-specific syntax and semantics. The plugin can be
    configured to use different OpenMP versions and can be enabled or disabled through
    the openmp option.

    Attributes
    ----------
    DEFAULT_VERSION : float
        The default OpenMP version to use if no specific version is requested.
    VERSION_MODULES : dict
        A mapping of OpenMP versions to their corresponding implementation modules.
    _options : dict
        Configuration options passed to the OpenMP plugin.

    See Also
    --------
    Plugin : Base class for all Pyccel plugins.
    OmpPatchRegistry : Registry for OpenMP patches applied to a single class.
    OmpPatchInfo : Information about a single OpenMP patch.

    Examples
    --------
    >>> from pyccel.plugins.Openmp.plugin import Openmp
    >>> plugin = Openmp()
    >>> plugin.set_options({'omp_version': 4.5, 'openmp': True})
    >>> 
    >>> # Register a parser with the plugin
    >>> class Parser:
    ...     def parse(self, code):
    ...         return code
    >>> parser = Parser()
    >>> plugin.register(parser)
    """
    __slots__ = ("_options",)

    CLI_OPTIONS = {
        'omp_version': {
            'choices': [4.5, 5.0],
            'type': float,
            'default': 4.5,
            'help': 'OpenMP version to use'
        },
        'openmp': {
            'action': 'store_true',
            'help': 'Enable OpenMP support'
        }
    }

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
        """
        Set options for the OpenMP plugin.

        This method configures the plugin with the provided options, which may
        affect how the plugin interacts with its targets. The options dictionary
        can include settings like the OpenMP version to use.

        Parameters
        ----------
        options : dict
            A dictionary of configuration options for the OpenMP plugin.
            Common options include 'omp_version' and 'accelerators'.

        See Also
        --------
        register : Register instances with the OpenMP plugin.
        refresh : Refresh all registered targets with current options.

        Examples
        --------
        >>> plugin = Openmp()
        >>> plugin.set_options({'omp_version': 4.5, 'openmp': True})
        """
        assert isinstance(options, dict)
        self._options.clear()
        self._options.update(options)

    def register(self, instances):
        """
        Register the provided instances with the OpenMP plugin.

        This method registers the given instances with the OpenMP plugin by creating
        patch registries for them and applying the appropriate OpenMP patches based
        on the configured version. It only applies patches if OpenMP is enabled in
        the accelerators option.

        Parameters
        ----------
        instances : list or object
            The instances to register with the OpenMP plugin. Can be a single object
            or a list of objects.

        See Also
        --------
        deregister : Deregister instances from the OpenMP plugin.
        refresh : Refresh all registered targets with current options.
        _apply_patches : Apply patches from the specified version module.
        """
        new_patch_regs = [OmpPatchRegistry(instance) for instance in instances
                            if not any(reg.target is instance for reg in self._patch_registries)]
        self._patch_registries += new_patch_regs
        if self._refresh:
            new_patch_regs = self._patch_registries
        version = self._resolve_version()
        if not self._options.get('openmp', False):
            return
        for reg in new_patch_regs:
            if version in reg.loaded_versions:
                continue
            self._apply_patches(version, reg)
            reg.loaded_versions.append(version)

    def refresh(self):
        """
        Refresh all registered targets with current options.

        This method reapplies patches to all registered targets using the current
        plugin options. It is useful when options have been changed and the changes
        need to be propagated to all targets.

        See Alsggo
        --------
        register : Register instances with the OpenMP plugin.
        deregister : Deregister instances from the OpenMP plugin.
        set_options : Set options for the OpenMP plugin.

        Examples
        --------
        >>> class Parser:
        ...     def parse(self, code):
        ...         return code
        >>> parser = Parser()
        >>> plugin = Openmp()
        >>> plugin.set_options({'openmp': True, 'omp_version': 4.5})
        >>> plugin.register(parser)
        >>> # Change options
        >>> plugin.set_options({'openmp': True, 'omp_version': 5.0})
        >>> # Refresh to apply new options
        >>> plugin.refresh()
        """
        self._refresh = True
        self.register(self.get_all_targets())
        self._refresh = False

    def deregister(self, instances):
        """
        Deregister the provided instances from the OpenMP plugin.

        This method removes the specified instances from the plugin by unloading
        all patches that were applied to them and removing their patch registries.

        Parameters
        ----------
        instances : list or object
            The instances to deregister from the OpenMP plugin. Can be a single object
            or a list of objects.

        See Also
        --------
        register : Register instances with the OpenMP plugin.
        refresh : Refresh all registered targets with current options.
        _unload_patches : Unload patches for a specific version.
        """
        targets = [reg for reg in self._patch_registries if reg.target in instances]
        for reg in targets:
            for ver in reg.loaded_versions:
                self._unload_patches(reg, ver)
        self._patch_registries = [reg for reg in self._patch_registries if reg not in targets]

    def _resolve_version(self):
        """
        Determine which OpenMP version to use based on options or environment.

        This method resolves the OpenMP version to use by checking the plugin options
        and environment variables. If no version is specified or the specified version
        is not supported, it falls back to the default version.

        Returns
        -------
        float
            The resolved OpenMP version to use.

        See Also
        --------
        _apply_patches : Apply patches from the specified version module.
        VERSION_MODULES: Dictionary mapping OpenMP versions to implementation modules.

        Examples
        --------
        >>> plugin = Openmp()
        >>> plugin.set_options({'omp_version': 4.5})
        >>> plugin._resolve_version()
        4.5
        >>> plugin.set_options({'omp_version': 999.0})  # Unsupported version
        >>> plugin._resolve_version() == plugin.DEFAULT_VERSION
        True
        """
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
        """
        Apply patches from the specified version module to parser classes.

        This method applies OpenMP-specific patches to the target parser class
        based on the specified OpenMP version. It finds the appropriate implementation
        module for the version, sets up the parser if needed, and applies patches
        to the parser's methods.

        Parameters
        ----------
        version : float
            The OpenMP version to use for patching.
        registry : OmpPatchRegistry
            The registry to which the patches will be added.

        See Also
        --------
        _unload_patches : Unload patches for a specific version.
        _resolve_version : Determine which OpenMP version to use.
        VERSION_MODULES: Dictionary mapping OpenMP versions to implementation modules.
        """
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
        """
        Unload patches for a specific OpenMP version.

        This method removes all patches associated with the specified OpenMP version
        from the target parser. It restores original methods or removes new methods
        added by the patches.

        Parameters
        ----------
        registry : OmpPatchRegistry
            The registry containing the patches to unload.
        version : float
            The OpenMP version for which to unload patches.

        See Also
        --------
        _apply_patches : Apply patches from the specified version module.
        deregister : Deregister instances from the OpenMP plugin.
        """
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
        registry.deregister_patches_for_version(version)
