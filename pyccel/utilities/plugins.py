"""Classes that handles plugins functionality"""
import importlib.util
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from pyccel.errors.errors import Errors
from pyccel.utilities.metaclasses import Singleton

errors = Errors()


@dataclass
class PatchInfo:
    """Store information about a single patch"""
    original_method: Optional[Callable]
    patched_method: Callable
    version: float
    method_name: str

    def __post_init__(self):
        if self.original_method is None:
            self.is_new_method = True
        else:
            self.is_new_method = False


@dataclass
class ClassPatchRegistry:
    """Registry for all patches applied to a single class"""
    target_class: type
    patches: Dict[str, List[PatchInfo]] = field(default_factory=dict)

    def register_patch(self, method_name, patch_info):
        """register a patch for a method"""
        if method_name not in self.patches:
            self.patches[method_name] = []
        self.patches[method_name].append(patch_info)

    def unregister_patch(self, method_name, patch_info):
        """unregister a patch for a method"""
        self.patches[method_name] = [pa for pa in self.patches[method_name] if pa is not patch_info]

    def get_original_method(self, method_name):
        """Get the original method before any patches were applied"""
        if method_name not in self.patches:
            return None
        return self.patches[method_name][0].original_method


class Plugin(ABC):
    """Abstract base class for Pyccel plugins."""

    def __init__(self):
        self._patch_registries = []
        self._setup_patch_registries()
        assert all(isinstance(reg, ClassPatchRegistry) for reg in self._patch_registries)

    @abstractmethod
    def handle_loading(self, options):
        """Handle loading plugin with provided options"""

    @abstractmethod
    def _setup_patch_registries(self):
        """Setup patch registries - override in subclasses to define patchable classes"""

    @property
    def name(self):
        """Return the plugin name, defaults to class name"""
        return self.__class__.__name__

    def get_patched_classes(self):
        """Return list of classes that have been patched by this plugin"""
        patched_classes = []
        for registry in self._patch_registries:
            patched_classes.append(registry.target_class)
        return patched_classes

    def _get_registry_for_class(self, target_class):
        """Get the registry for a specific class"""
        for registry in self._patch_registries:
            if registry.target_class == target_class:
                return registry
        return None

    def get_patched_methods(self, target_class):
        """Return list of method names patched in the given class"""
        registry = self._get_registry_for_class(target_class)
        if not registry:
            return []
        return list(registry.patches.keys())

    def get_original_method(self, target_class, method_name):
        """Return the original method before patching (for testing)"""
        registry = self._get_registry_for_class(target_class)
        if not registry:
            return None
        return registry.get_original_method(method_name)


class Plugins(metaclass=Singleton):
    """Manager for Pyccel plugins"""

    __slots__ = ("_plugins",)

    def __init__(self, plugins_dir=None):
        self._plugins = {}
        self._load_plugins(plugins_dir)

    def _load_plugins(self, plugins_dir=None):
        """Discover and load all plugins from the plugins directory"""
        if plugins_dir is None:
            current_dir = os.path.dirname(__file__)
            plugins_dir = os.path.abspath(os.path.join(current_dir, "..", "plugins"))

        if not os.path.isdir(plugins_dir):
            errors.report(
                f"Plugin directory not found: {plugins_dir}.",
                severity='warning')
            return

        plugin_folders = [
            d for d in os.listdir(plugins_dir)
            if os.path.isdir(os.path.join(plugins_dir, d)) and not d.startswith('_')
        ]

        for folder in plugin_folders:
            try:
                plugin = self._load_plugin_from_folder(plugins_dir, folder)
                if plugin:
                    self._plugins[plugin.name] = plugin
            except (ImportError, FileNotFoundError, ValueError) as e:
                errors.report(
                    f"Error loading plugin '{folder}': {str(e)}",
                    severity='warning')

    def _load_plugin_from_folder(self, plugins_dir, folder):
        """Load a single plugin from the specified folder"""
        plugin_path = os.path.join(plugins_dir, folder, "plugin.py")

        if not os.path.exists(plugin_path):
            raise FileNotFoundError(f"Plugin file not found at {plugin_path}")

        # import the module
        module_name = f"{folder}_pyccel_plugin"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin: {plugin_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # find plugin class in the module
        plugin_class = None
        for _, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    issubclass(obj, Plugin) and
                    obj.__module__ == module.__name__ and
                    obj is not Plugin):
                plugin_class = obj
                break

        if plugin_class is None:
            raise ValueError(f"No valid plugin class found in {plugin_path}")

        return plugin_class()

    def handle_loading(self, options):
        """Load all available plugins with the provided options"""
        for plugin_name, plugin in self._plugins.items():
            try:
                plugin.handle_loading(options)
            # Catching all exceptions because plugin loading may fail in unpredictable ways
            except Exception as e:  # pylint: disable=broad-exception-caught
                # plugin.handle_loading({'clear':True})
                errors.report(
                    f"Error in plugin '{plugin_name}' during loading: {str(e)}",
                    severity='warning')

    def get_plugin(self, name):
        """Get a plugin by name"""
        return self._plugins.get(name, None)

    def get_all_plugins(self):
        """Get all loaded plugins"""
        return list(self._plugins.values())
