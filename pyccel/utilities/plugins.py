"""Classes that handles plugins functionality"""
import importlib.util
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from pyccel.errors.errors import Errors
from pyccel.utilities.metaclasses import Singleton

errors = Errors()


@dataclass
class PatchInfo:
    """Store information about a single patch"""
    original_method: Optional[Callable]
    patched_method: Callable
    method_name: str

    def __post_init__(self):
        if self.original_method is None:
            self.is_new_method = True
        else:
            self.is_new_method = False


@dataclass
class PatchRegistry:
    """Registry for all patches applied to a single class"""
    target: Any
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
        assert all(isinstance(reg, PatchRegistry) for reg in self._patch_registries)

    @abstractmethod
    def register(self, instances, refresh=False):
        """Handle loading plugin with provided options"""

    @abstractmethod
    def unregister(self, instances):
        """Handle loading plugin with provided options"""

    @abstractmethod
    def set_options(self, options):
        """Handle loading plugin with provided options"""

    @property
    def name(self):
        """Return the plugin name, defaults to class name"""
        return self.__class__.__name__

    def get_registry_for(self, target):
        """Get the registry for a specific class"""
        return next((registry for registry in self._patch_registries if registry.target is target), None)

    def is_registered(self, target):
        return any(registry.target is target for registry in self._patch_registries)

    def get_all_targets(self):
        """Get all objects targeted by the plugin"""
        return list(set(reg.target for reg in self._patch_registries))

    def get_patched_methods(self, target):
        """Return list of method names patched in the given class"""
        registry = self.get_registry_for(target)
        if not registry:
            return []
        return list(registry.patches.keys())

    def get_original_method(self, target, method_name):
        """Return the original method before patching (for testing)"""
        registry = self.get_registry_for(target)
        if not registry:
            return None
        return registry.get_original_method(method_name)


class Plugins(metaclass=Singleton):
    """Manager for Pyccel plugins"""

    __slots__ = ("_plugins", "_options")

    def __init__(self, plugins_dir=None):
        self._plugins = []
        self._options = {}
        self.load_plugins(plugins_dir)

    def set_options(self, options, refresh=False):
        assert isinstance(options, dict)
        self._options = options
        plugins = self._plugins
        for plugin in plugins:
            plugin.set_options(options)
            if refresh:
                plugin.register((), refresh)

    def load_plugins(self, plugins_dir=None):
        """Discover and load all plugins from the plugins directory"""
        self.unload_plugins()
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
                    self._plugins.append(plugin)
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

    def unload_plugins(self):
        for plugin in self._plugins:
            self.unregister(plugin.get_all_targets(), (plugin,))
        self._plugins = []

    def register(self, instances, refresh=False, plugins = ()):
        """Register the given instances """
        if not plugins:
            plugins = self._plugins
        for plugin in plugins:
            try:
                plugin.register(instances, refresh=refresh)
            # Catching all exceptions because plugin loading may fail in unpredictable ways
            except Exception as e:  # pylint: disable=broad-exception-caught
                # plugin.handle_loading({'clear':True})
                errors.report(
                    f"Error in plugin '{plugin.name}' during loading: {str(e)}",
                    severity='warning')
                raise e

    def unregister(self, instances, plugins = ()):
        if not plugins:
            plugins = self._plugins
        for plugin in plugins:
            plugin.unregister(instances)

    def get_plugin(self, name):
        """Get a plugin by name"""
        return next((p for p in self._plugins if p.name == name), None)

    def get_plugins(self):
        return self._plugins

    def set_plugins(self, plugins):
        self.unload_plugins()
        self._plugins = plugins
