import pluggy
from . import hookspecs
from . import LineAnnot

def get_plugin_manager():
    pm = pluggy.PluginManager("pyccel")

    # Register expected hook format
    pm.add_hookspecs(hookspecs)

    # Search for available plugins on the system
    pm.load_setuptools_entrypoints("pyccel")

    # Register plugins provided inside Pyccel
    pm.register(LineAnnot.plugin, 'line_annotation')
    #pm.register(openmp, 'openmp')

    return pm

def get_plugin_cli_options(plugin_manager, parser, cli_tool):
    group = parser.add_argument_group("Plugins")
    for plugin in plugin_manager.get_plugins():
        group.add_argument(
            "--"+plugin_manager.get_name(plugin),
            action="store_true",
            help=plugin.get_description()
        )

        plugin.add_cli_options(parser=group, cli_tool=cli_tool)

    parser.set_defaults(plugin_manager = plugin_manager)

def deactivate_plugins(plugin_manager, active_plugins):
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        if name in active_plugins:
            if not active_plugins[name]:
                plugin_manager.unregister(plugin)
            active_plugins.pop(name)
        else:
            plugin_manager.unregister(plugin)

def get_syntactic_class(plugin_manager, BaseClass):
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_syntactic_methods()
        except AttributeError:
            continue

        BaseClass = type(name + BaseClass.__name__,
                         (BaseClass,),
                         {m.__name__: m for m in new_methods})

    return BaseClass

def get_semantic_class(plugin_manager, BaseClass):
    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_semantic_methods()
        except AttributeError:
            continue

        BaseClass = type(name + BaseClass.__name__,
                         (BaseClass,),
                         {m.__name__: m for m in new_methods})

    return BaseClass

def get_codegen_class(plugin_manager, BaseClass, language):
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                BaseClass = get_codegen_class(language)
            except AttributeError:
                continue
            break

    if BaseClass is None:
        raise ValueError(f"{language} language is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_codegen_methods(language)
        except AttributeError:
            continue

        BaseClass = type(name + BaseClass.__name__,
                         (BaseClass,),
                         {m.__name__: m for m in new_methods})

    return BaseClass

def get_wrapper_class(plugin_manager, BaseClass, start_language, target_language):
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                BaseClass = get_wrapper_class(start_language, target_language)
            except AttributeError:
                continue
            break

    if BaseClass is None:
        raise ValueError(f"Wrapping from {start_language} to {target_language} is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_wrapper_methods(start_language, target_language)
        except AttributeError:
            continue

        BaseClass = type(name + BaseClass.__name__,
                         (BaseClass,),
                         {m.__name__: m for m in new_methods})

    return BaseClass

def get_build_generation_class(plugin_manager, BaseClass, build_gen_method):
    if BaseClass is None:
        for plugin in plugin_manager.get_plugins():
            try:
                BaseClass = get_build_generation_class(build_gen_method)
            except AttributeError:
                continue
            break

    if BaseClass is None:
        raise ValueError(f"{build_gen_method} build generation method is not available")

    for plugin in plugin_manager.get_plugins():
        name = plugin_manager.get_name(plugin)
        try:
            new_methods = plugin.get_updated_build_generation_methods(build_gen_method)
        except AttributeError:
            continue

        BaseClass = type(name + BaseClass.__name__,
                         (BaseClass,),
                         {m.__name__: m for m in new_methods})

    return BaseClass
