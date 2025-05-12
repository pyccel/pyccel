import os
import importlib.util
import inspect
from abc import ABC, abstractmethod
from pyccel.utilities.metaclasses import Singleton


class Plugin(ABC):
    @abstractmethod
    def handle_loading(self, options):
        pass


class Plugins(metaclass=Singleton):
    def __init__(self, plugins_dir=None):
        if plugins_dir is None:
            current_dir = os.path.dirname(__file__)
            plugins_dir = os.path.abspath(os.path.join(current_dir, "..", "plugins"))

        folders = [d for d in os.listdir(plugins_dir)
                          if os.path.isdir(os.path.join(plugins_dir, d)) and not d.startswith('_')]
        self._plugins = []


        for f in folders:
            plugin_path = os.path.join(plugins_dir, f, "plugin.py")

            if not os.path.exists(plugin_path):
                raise ValueError(f"Plugin {f} not found at {plugin_path}")

            module_name = f"{f}_pyccel_plugin"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)

            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load plugin: {plugin_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Plugin) and obj.__module__ == module.__name__:
                    plugin_class = obj
                    break

            if plugin_class is None:
                raise ValueError(f"No valid plugin class found in {plugin_path}")

            plugin_instance = plugin_class()
            self._plugins.append(plugin_instance)

    def handle_loading(self, options):
        """Load all available extensions"""

        for plugin in self._plugins:
            plugin.handle_loading(options)
