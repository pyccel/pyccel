import pluggy
from . import hookspecs
from . import LineAnnot

def get_plugin_manager():
    pm = pluggy.PluginManager("pyccel")

    # Register exected hook format
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
