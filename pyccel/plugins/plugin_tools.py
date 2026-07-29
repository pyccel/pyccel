"""
Utility functions for managing pyccel plugins.

This module provides methods for creating the plugin manager, and calling the
methods implemented as hooks in plugins.
"""

import pluggy

from pyccel.parser.semantic import SemanticParser
from pyccel.parser.syntactic import SyntaxParser

from . import LineAnnot, Openmp, hookspecs


def get_plugin_manager():
    """
    Create a plugin manager with all available pyccel plugins registered.

    The manager registers the built-in hook specifications, loads any
    third-party plugins published as `pyccel` entry points, and then
    registers the plugins that ship with pyccel itself.

    Returns
    -------
    pluggy.PluginManager
        A fully initialised plugin manager ready for use.
    """
    pm = pluggy.PluginManager("pyccel")

    # Register expected hook format
    pm.add_hookspecs(hookspecs)

    # Search for available plugins on the system
    pm.load_setuptools_entrypoints("pyccel")

    # Register plugins provided inside Pyccel
    pm.register(LineAnnot.plugin, "line_annotation")
    pm.register(Openmp.plugin, "openmp")

    return pm


def get_plugin_cli_options(plugin_manager, parser, cli_tool):
    """
    Add a "Plugins" argument group to parser populated by all registered plugins.

    Each plugin gets an `--<name>` flag that enables it, followed by any
    plugin-specific options contributed via the `add_cli_options` hook.
    The plugin manager is stored in parser via the `plugin_manager` default,
    so it is available to the command that processes the parsed arguments.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager whose registered plugins will contribute options.
    parser : argparse.ArgumentParser
        The argument parser to which the plugin argument group will be added.
    cli_tool : str
        The name of the CLI tool being invoked.
    """
    group = parser.add_argument_group("Plugins")
    for plugin in plugin_manager.get_plugins():
        group.add_argument(
            "--" + plugin_manager.get_name(plugin),
            action="store_true",
            help=plugin.get_description(),
        )

        plugin.add_cli_options(parser=parser, cli_tool=cli_tool)

    parser.set_defaults(plugin_manager=plugin_manager)


def handle_plugin_arguments(plugin_manager, kwargs):
    """
    Use kwargs to unregister unused plugins and collect arguments.

    The arguments are used to recognise which plugins are registered. This is
    those that are present and associated with a truthy value. Note that a
    plugin may not appear in kwargs if epyccel is used. Unused plugins are
    unregistered. The kwargs are passed to the remaining plugins to initialise
    any options.
    Plugin flags are removed from kwargs to ensure that the rest of the code
    doesn't see plugin arguments.
    The flags are provided either via the CLI interface (via the options provided
    by get_plugin_cli_options), or via epyccel arguments (but the user has to
    know what arguments are available).

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing all plugins.
    kwargs : dict
        The keyword arguments passed to the pipeline.
    """
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        if name not in kwargs:
            plugin_manager.unregister(plugin)
            plugin.remove_cli_arguments(kwargs)
        elif not kwargs[name]:
            plugin_manager.unregister(plugin)
            kwargs.pop(name)
            plugin.remove_cli_arguments(kwargs)
        else:
            kwargs.pop(name)
            plugin.read_cli_arguments(kwargs)


def get_syntactic_class(plugin_manager):
    """
    Return a syntactic parser subclass augmented with methods from active plugins.

    For each plugin that implements `get_updated_syntactic_methods`, a new
    subclass of SyntaxParser is created with the plugin's methods injected.
    Plugins are applied in registration order, so later plugins can override
    earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.

    Returns
    -------
    type
        A (possibly new) class derived from SyntaxParser augmented with the plugin methods.
    """
    BaseClass = SyntaxParser
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_syntactic_methods()
        except AttributeError:
            continue

        BaseClass = type(
            name + BaseClass.__name__,
            (BaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return BaseClass


def get_semantic_class(plugin_manager):
    """
    Return a semantic parser subclass augmented with methods from active plugins.

    For each plugin that implements `get_updated_semantic_methods`, a new
    subclass of SemanticParser is created with the plugin's methods injected.
    Plugins are applied in registration order, so later plugins can override
    earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.

    Returns
    -------
    type
        A (possibly new) class derived from BaseClass augmented with the plugin methods.
    """
    BaseClass = SemanticParser
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_semantic_methods()
        except AttributeError:
            continue

        BaseClass = type(
            name + BaseClass.__name__,
            (BaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return BaseClass


def get_codegen_class(plugin_manager, BaseClass, language):
    """
    Return a code-generation class for language augmented with plugin methods.

    If BaseClass is `None`, the plugins are used to find an implementation of a
    code-generation class for the requested language via the `get_codegen_class`
    hook. A `ValueError` is raised if no base class can be found.

    For each plugin that implements `get_updated_codegen_methods`, a new
    subclass of BaseClass is created with the plugin's methods injected. Plugins
    are applied in registration order, so later plugins can override earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.
    BaseClass : type or None
        The base code-generation class, or `None` to discover one from
        the registered plugins.
    language : str
        The target language (e.g. 'c', 'fortran').

    Returns
    -------
    type
        A (possibly new) class derived from CodePrinter augmented with the plugin methods.

    Raises
    ------
    ValueError
        If BaseClass is `None` and no plugin provides a class for language.
    """
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                NewBaseClass = plugin.get_codegen_class(language)
            except AttributeError:
                continue
            else:
                # Stop searching for a codegen class at the first plugin providing one
                break
    else:
        NewBaseClass = BaseClass

    if NewBaseClass is None:
        raise ValueError(f"{language} language is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_codegen_methods(language)
        except AttributeError:
            continue

        NewBaseClass = type(
            name + NewBaseClass.__name__,
            (NewBaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return NewBaseClass


def get_wrapper_class(plugin_manager, BaseClass, start_language, target_language):
    """
    Return a wrapper class for wrapping code in start_language to the target_language.

    If BaseClass is `None`, the plugins are used to find an implementation of a
    wrapper class for the requested languages via the `get_wrapper_class`
    hook. A `ValueError` is raised if no base class can be found.
    In this case the target_language will also be None, its value is ascertained from
    `get_wrapper_class`.

    For each plugin that implements `get_updated_wrapper_methods`, a new
    subclass of BaseClass is created with the plugin's methods injected. Plugins
    are applied in registration order, so later plugins can override earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.
    BaseClass : type or None
        The base wrapper class, or `None` to discover one from the
        registered plugins.
    start_language : str
        The source language of the wrapper (e.g. 'fortran').
    target_language : str or None
        The target language of the wrapper (e.g. 'python').

    Returns
    -------
    NewBaseClass : type
        A (possibly new) class derived from BaseClass augmented with the plugin methods.
    target_language : str
        The target language of the wrapper (e.g. 'python'). This is equal to the input
        target language unless the input was None, then it is the target language of the
        wrapper class returned by `get_wrapper_class`.

    Raises
    ------
    ValueError
        If BaseClass is `None` and no plugin provides a suitable wrapper class.
    """
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                NewBaseClass, target_language = plugin.get_wrapper_class(start_language)
            except (AttributeError, TypeError):
                continue
            else:
                # Stop searching for a wrapper class at the first plugin providing one
                break
    else:
        NewBaseClass = BaseClass

    if NewBaseClass is None:
        raise ValueError(
            f"Wrapping from {start_language} to {target_language} is not available"
        )

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_wrapper_methods(
                start_language, target_language
            )
        except AttributeError:
            continue

        NewBaseClass = type(
            name + NewBaseClass.__name__,
            (NewBaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return NewBaseClass, target_language


def get_wrapper_codegen_class(plugin_manager, BaseClass, language):
    """
    Return a code-generation class for printing the wrapper in the language.

    If BaseClass is `None`, the plugins are used to find an implementation of a
    code-generation class for the requested language via the `get_codegen_class`
    hook. A `ValueError` is raised if no base class can be found.

    For each plugin that implements `get_updated_codegen_methods`, a new
    subclass of BaseClass is created with the plugin's methods injected. Plugins
    are applied in registration order, so later plugins can override earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.
    BaseClass : type or None
        The base code-generation class, or `None` to discover one from
        the registered plugins.
    language : str
        The target language (e.g. 'c', 'fortran').

    Returns
    -------
    type
        A (possibly new) class derived from CodePrinter augmented with the plugin methods.

    Raises
    ------
    ValueError
        If BaseClass is `None` and no plugin provides a class for language.
    """
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                NewBaseClass = plugin.get_wrapper_codegen_class(language)
            except AttributeError:
                continue
            else:
                # Stop searching for a wrapper codegen class at the first plugin providing one
                break
    else:
        NewBaseClass = BaseClass

    if NewBaseClass is None:
        raise ValueError(f"{language} language is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_wrapper_codegen_methods(language)
        except AttributeError:
            continue

        NewBaseClass = type(
            name + NewBaseClass.__name__,
            (NewBaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return NewBaseClass


def get_build_generation_class(plugin_manager, BaseClass, build_gen_method):
    """
    Return a build-generation class for build_gen_method augmented with plugin methods.

    If BaseClass is `None`, the plugins are used to find an implementation of a
    build-generation class for the requested method via the `get_build_generation_class`
    hook. A `ValueError` is raised if no base class can be found.

    For each plugin that implements `get_updated_build_generation_methods`, a new
    subclass of BaseClass is created with the plugin's methods injected. Plugins
    are applied in registration order, so later plugins can override earlier ones.

    Parameters
    ----------
    plugin_manager : pluggy.PluginManager
        The plugin manager containing the registered plugins.
    BaseClass : type or None
        The base build-generation class, or `None` to discover one from
        the registered plugins.
    build_gen_method : str
        The name of the build generation method (e.g. 'cmake').

    Returns
    -------
    type
        A (possibly new) class derived from BaseClass augmented with the plugin methods.

    Raises
    ------
    ValueError
        If BaseClass is `None` and no plugin provides a class for build_gen_method.
    """
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                NewBaseClass = plugin.get_build_generation_class(build_gen_method)
            except AttributeError:
                continue
            else:
                # Stop searching for a build generation class at the first plugin providing one
                break
    else:
        NewBaseClass = BaseClass

    if NewBaseClass is None:
        raise ValueError(f"{build_gen_method} build generation method is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_build_generation_methods(build_gen_method)
        except AttributeError:
            continue

        NewBaseClass = type(
            name + NewBaseClass.__name__,
            (NewBaseClass,),
            {m.__name__: m for m in new_methods},
        )

    return NewBaseClass
