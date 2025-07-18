"""Classes that handle plugins functionality."""
import importlib.util
import inspect
import os
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from pyccel.errors.errors import Errors
from pyccel.utilities.metaclasses import Singleton
from pyccel.errors.messages import PLUGIN_DIRECTORY_NOT_FOUND

__all__ = (
    "add_plugin_arguments",
    "collect_plugin_options",
    "PatchInfo",
    "PatchRegistry",
    "Plugin",
    "Plugins",
)

errors = Errors()


def add_plugin_arguments(parser):
    """
    Discover and add plugin arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add plugin arguments to.

    """
    added_options = []

    plugins_manager = Plugins()
    plugins = plugins_manager.get_plugins()

    for plugin in plugins:
        if hasattr(plugin.__class__, 'CLI_OPTIONS'):
            options = plugin.__class__.CLI_OPTIONS
            plugin_name = plugin.name

            group = parser.add_argument_group(f'{plugin_name} Options')
            for option_name, option_config in options.items():
                if option_name in added_options:
                    errors.report(
                        f"Option '{option_name}' already added by another plugin, skipping for plugin {plugin_name}",
                        severity='warning')
                    continue

                flag = f'--{option_name.replace("_", "-")}'
                try:
                    group.add_argument(flag, dest=option_name, **option_config)
                    added_options.append(option_name)
                except argparse.ArgumentError as e:
                    errors.report(
                        f"Argument conflict for '{flag}' in plugin {plugin_name}: {e}",
                        severity='warning')


def collect_plugin_options(args):
    """
    Collect all plugin options from parsed arguments into a single dictionary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict
        A dictionary containing all plugin options.
    """
    options = {}
    option_names = []
    plugins_manager = Plugins()
    plugins = plugins_manager.get_plugins()
    for plugin in plugins:
        if hasattr(plugin.__class__, 'CLI_OPTIONS'):
            option_names.extend(plugin.__class__.CLI_OPTIONS.keys())

    for option_name in option_names:
        if hasattr(args, option_name):
            options[option_name] = getattr(args, option_name)
    return options

@dataclass
class PatchInfo:
    """
    Store information about a single patch.

    This class is used to store information about a method patch, including
    the original method, the patched method, and the method name. It also
    determines whether the patch is for a new method or an existing one.

    Parameters
    ----------
    original_method : Optional[Callable]
        The original method that is being patched. If None, the patch
        is considered to be for a new method.
    patched_method : Callable
        The new method that will replace the original method.
    method_name : str
        The name of the method being patched.

    See Also
    --------
    PatchRegistry : Registry for patches applied to a single class.

    Examples
    --------
    >>> def original_func(x):
    ...     return x
    >>> def patched_func(x):
    ...     return x * 2
    >>> patch = PatchInfo(original_func, patched_func, "func")
    """
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
    """
    Registry for all patches applied to a single class.

    This class serves as a container for all patches that are applied to a 
    specific target class. It maintains a dictionary of patches organized by 
    method name.

    Parameters
    ----------
    target : Any
        The target class or object to which patches will be applied.
    patches : Dict[str, List[PatchInfo]], optional
        A dictionary mapping method naming to lists of PatchInfo objects.
        Default is an empty dictionary.

    See Also
    --------
    PatchInfo : Store information about a single patch.

    Examples
    --------
    >>> class MyClass:
    ...     def method(self):
    ...         return 1
    >>> obj = MyClass()
    >>> registry = PatchRegistry(obj)
    >>> def patched_method(self):
    ...     return 2
    >>> patch_info = PatchInfo(obj.method, patched_method, "method")
    >>> registry.register_patch("method", patch_info)
    """
    target: Any
    patches: Dict[str, List[PatchInfo]] = field(default_factory=dict)

    def register_patch(self, method_name, patch_info):
        """
        Register a patch for a method.

        This method adds a new patch to the registry for the specified method name.
        If the method name is not already in the registry, a new entry is created.

        Parameters
        ----------
        method_name : str
            The name of the method to be patched.
        patch_info : PatchInfo
            The patch information object containing the original and patched methods.

        See Also
        --------
        PatchInfo : Store information about a single patch.

        Examples
        --------
        >>> class MyClass:
        ...     def method(self):
        ...         return 1
        >>> obj = MyClass()
        >>> registry = PatchRegistry(obj)
        >>> def method(self):
        ...     return 2
        >>> patch_info = PatchInfo(obj.method, method, "method")
        >>> registry.register_patch("method", patch_info)
        >>> len(registry.patches["method"])
        1
        """
        if method_name not in self.patches:
            self.patches[method_name] = []
        self.patches[method_name].append(patch_info)


class Plugin(ABC):
    """
    Abstract base class for Pyccel plugins.

    This class defines the interface that all Pyccel plugins must implement.
    Plugins are used to extend the functionality of Pyccel by providing
    additional features or modifying existing behavior through method patching.

    Each plugin maintains a list of patch registries, which track the patches
    applied to different target classes or objects.

    See Also
    --------
    PatchRegistry : Registry for all patches applied to a single class.
    PatchInfo : Store information about a single patch.
    Plugins : Manager for Pyccel plugins.
    """

    def __init__(self):
        self._patch_registries = []
        assert all(isinstance(reg, PatchRegistry) for reg in self._patch_registries)

    @abstractmethod
    def register(self, instances):
        """
        Register instances with this plugin.

        This method is called to register instances with the plugin.
        Implementations should apply the necessary patches to the provided instances.

        Parameters
        ----------
        instances : list or object
            The instances to register with this plugin. Can be a single object
            or a list of objects.

        See Also
        --------
        deregister : Deregister instances from this plugin.
        refresh : Refresh all registered targets.
        """

    @abstractmethod
    def refresh(self):
        """
        Refresh all registered targets.

        This method is called to refresh all targets that have been registered
        with this plugin. It should reapply patches or apply new patches if necessary
        to ensure that the plugin's functionality is up to date.

        See Also
        --------
        register : Register instances with this plugin.
        deregister : Deregister instances from this plugin.
        """

    @abstractmethod
    def deregister(self, instances):
        """
        Deregister instances from this plugin.

        This method is called to deregister instances from the plugin.
        Implementations should remove all patches applied to the
        provided instances.

        Parameters
        ----------
        instances : list or object
            The instances to deregister from this plugin. Can be a single object
            or a list of objects.

        See Also
        --------
        register : Register instances with this plugin.
        refresh : Refresh all registered targets.
        """

    @abstractmethod
    def set_options(self, options):
        """
        Set options for this plugin.

        This method is called to configure the plugin with the provided options.
        Implementations should update their behavior based on the options.

        Parameters
        ----------
        options : dict
            A dictionary of options to configure the plugin.

        See Also
        --------
        register : Register instances with this plugin.
        refresh : Refresh all registered targets.
        """

    @property
    def name(self):
        """
        Return the plugin name, defaults to class name.

        This property returns the name of the plugin, which by default is the
        name of the plugin class. This can be used to identify the plugin in
        logs or when retrieving it from the Plugins manager.

        See Also
        --------
        Plugins.get_plugin : Get a plugin by name.
        """
        return self.__class__.__name__

    def is_registered(self, target):
        """
        Check if a target is registered with this plugin.

        This method determines whether the specified target object has been
        registered with this plugin. It checks if the target exists in any
        of the plugin's patch registries.

        Parameters
        ----------
        target : object
            The target object to check for registration.

        Returns
        -------
        bool
            True if the target is registered with this plugin, False otherwise.

        See Also
        --------
        register : Register instances with this plugin.
        deregister : Deregister instances from this plugin.
        get_all_targets : Get all objects targeted by the plugin.
        """
        return any(registry.target is target for registry in self._patch_registries)

    def get_all_targets(self):
        """
        Get all objects targeted by the plugin.

        This method returns a list of all target objects that have been
        registered with this plugin. It collects all targets from the
        plugin's patch registries and returns them as a unique list.

        Returns
        -------
        list
            A list of all target objects registered with this plugin.

        See Also
        --------
        is_registered : Check if a target is registered with this plugin.
        register : Register instances with this plugin.
        deregister : Deregister instances from this plugin.
        """
        return list(set(reg.target for reg in self._patch_registries))


class Plugins(metaclass=Singleton):
    """
    Manager for Pyccel plugins.

    This class is responsible for discovering, loading, and managing plugins
    for Pyccel. It is implemented as a singleton to ensure that there is only
    one instance of the plugin manager throughout the application.

    The Plugins class provides methods for loading plugins from a directory,
    registering and deregistering instances with plugins, and retrieving
    plugins by name.

    Parameters
    ----------
    plugins_dir : str, optional
        The directory from which to load plugins. If not provided, the default
        plugins directory will be used.

    See Also
    --------
    Plugin : Abstract base class for Pyccel plugins.
    PatchRegistry : Registry for all patches applied to a single class.
    PatchInfo : Store information about a single patch.
    """

    __slots__ = ("_plugins", "_options")

    def __init__(self, plugins_dir=None):
        self._plugins = []
        self._options = {}
        self.load_plugins(plugins_dir)

    def set_options(self, options, refresh=False):
        """
        Set options for all plugins.

        This method sets the provided options for all loaded plugins. It can
        also refresh all plugins to ensure that the new options take effect
        immediately.

        Parameters
        ----------
        options : dict
            A dictionary of options to configure the plugins.
        refresh : bool, optional
            Whether to refresh all plugins after setting the options.
            Default is False.

        See Also
        --------
        Plugin.set_options : Set options for a specific plugin.
        refresh : Refresh all registered targets.

        Examples
        --------
        >>> plugins = Plugins()
        >>> plugins.load_plugins()
        >>> plugins.set_options({"option1": "value1", "option2": True}, refresh=True)
        """
        assert isinstance(options, dict)
        self._options = options
        plugins = self._plugins
        for plugin in plugins:
            plugin.set_options(options)
            if refresh:
                plugin.refresh()

    def load_plugins(self, plugins_dir=None):
        """
        Discover and load all plugins from the plugins directory.

        This method searches for and loads all plugins from the specified
        directory. If no directory is provided, it uses the default plugins
        directory. Each plugin is loaded by importing its plugin.py file and
        instantiating the plugin class.

        Parameters
        ----------
        plugins_dir : str, optional
            The directory from which to load plugins. If not provided, the default
            plugins directory will be used.

        See Also
        --------
        _load_plugin_from_folder : Load a single plugin from a folder.
        unload_plugins : Unload all plugins.

        Examples
        --------
        >>> plugins = Plugins()
        >>> plugins.load_plugins("/path/to/plugins")
        """
        self.unload_plugins()
        if plugins_dir is None:
            current_dir = os.path.dirname(__file__)
            plugins_dir = os.path.abspath(os.path.join(current_dir, "..", "plugins"))

        if not os.path.isdir(plugins_dir):
            errors.report(
                PLUGIN_DIRECTORY_NOT_FOUND.format(plugins_dir),
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
        """
        Load a single plugin from the specified folder.

        This method loads a plugin from a specific folder within the plugins
        directory. It looks for a plugin.py file in the folder, imports it,
        and instantiates the plugin class found in the module.

        Parameters
        ----------
        plugins_dir : str
            The root directory containing all plugin folders.
        folder : str
            The name of the specific folder containing the plugin to load.

        Returns
        -------
        Plugin or None
            An instance of the plugin class if found and successfully loaded,
            or None if the plugin could not be loaded.

        See Also
        --------
        load_plugins : Discover and load all plugins from the plugins directory.
        """
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
        """
        Unload all plugins and deregister all their targets.

        This method unloads all currently loaded plugins and deregisters all
        targets registered with those plugins. This effectively
        removes all patches applied by the plugins.

        See Also
        --------
        load_plugins : Discover and load all plugins from the plugins directory.
        deregister : Deregister instances from plugins.

        Examples
        --------
        >>> plugins = Plugins()
        >>> plugins.load_plugins()
        >>> # Do some work with plugins
        >>> plugins.unload_plugins()
        """
        for plugin in self._plugins:
            self.deregister(plugin.get_all_targets(), (plugin,))
        self._plugins = []

    def register(self, instances, plugins = ()):
        """
        Register the given instances with plugins.

        This method registers the provided instances with the specified plugins,
        or with all loaded plugins if none are specified. It applies the
        necessary patches to the instances to enable the plugin functionality.

        Parameters
        ----------
        instances : list or object
            The instances to register with the plugins. Can be a single object
            or a list of objects.
        plugins : tuple or list, optional
            The plugins to register the instances with. If not provided, all
            loaded plugins will be used. Default is an empty tuple.

        See Also
        --------
        deregister : Deregister instances from plugins.
        Plugin.register : Register instances with a specific plugin.
        """
        if not plugins:
            plugins = self._plugins
        for plugin in plugins:
            try:
                plugin.register(instances)
            # Catching all exceptions because plugin loading may fail in unpredictable ways
            except Exception as e:  # pylint: disable=broad-exception-caught
                # plugin.handle_loading({'clear':True})
                errors.report(
                    f"Error in plugin '{plugin.name}' during loading: {str(e)}",
                    severity='warning')
                raise e

    def deregister(self, instances, plugins = ()):
        """
        Deregister the given instances from the given plugins.

        This method deregisters the provided instances from the specified plugins,
        or from all loaded plugins if none are specified. It removes all patches
        applied to the instances by the plugins.

        Parameters
        ----------
        instances : list or object
            The instances to deregister from the plugins. Can be a single object
            or a list of objects.
        plugins : tuple or list, optional
            The plugins to deregister the instances from. If not provided, all
            loaded plugins will be used. Default is an empty tuple.

        See Also
        --------
        register : Register instances with plugins.
        Plugin.deregister : Deregister instances from a specific plugin.
        """
        if not plugins:
            plugins = self._plugins
        for plugin in plugins:
            plugin.deregister(instances)

    def get_plugin(self, name):
        """
        Get a plugin by name.

        This method retrieves a plugin instance by its name. If no plugin
        with the given name is found, it returns None.

        Parameters
        ----------
        name : str
            The name of the plugin to retrieve.

        Returns
        -------
        Plugin or None
            The plugin instance if found, or None if no plugin with the
            given name exists.

        See Also
        --------
        get_plugins : Get all plugins.
        """
        return next((p for p in self._plugins if p.name == name), None)

    def get_plugins(self):
        """
        Get all loaded plugins.

        This method returns a list of all plugin instances that have been
        loaded by the plugin manager.

        Returns
        -------
        list
            A list of all loaded plugin instances.

        See Also
        --------
        get_plugin : Get a plugin by name.
        load_plugins : Discover and load all plugins from the plugins directory.
        """
        return self._plugins

    def set_plugins(self, plugins):
        """
        Set the list of plugin instances.

        This method replaces the current list of plugin instances with a new one.
        It first unloads all existing plugins before setting the new ones.

        Parameters
        ----------
        plugins : list
            A list of plugin instances to set as the current plugins.

        See Also
        --------
        get_plugins : Get all loaded plugins.
        load_plugins : Discover and load all plugins from the plugins directory.
        unload_plugins : Unload all plugins and deregister all their targets.
        """
        self.unload_plugins()
        self._plugins = plugins
