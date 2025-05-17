import importlib.util
import importlib.util
import inspect
import os
from abc import ABC, abstractmethod

from pyccel.errors.errors import Errors
from pyccel.utilities.metaclasses import Singleton

errors = Errors()


class Plugin(ABC):
    """Abstract base class for Pyccel plugins."""

    @abstractmethod
    def handle_loading(self, options):
        """Handle loading plugin with provided options"""
        pass

    @property
    def name(self):
        """Return the plugin name, defaults to class name"""
        return self.__class__.__name__


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
            except Exception as e:
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
        for name, obj in inspect.getmembers(module):
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
            except Exception as e:
                errors.report(
                    f"Error in plugin '{plugin_name}' during loading: {str(e)}",
                    severity='warning')

    def get_plugin(self, name):
        """Get a plugin by name"""
        return self._plugins.get(name, None)

    def get_all_plugins(self):
        """Get all loaded plugins"""
        return list(self._plugins.values())
