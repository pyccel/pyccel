import pluggy
from . import hookspecs

def get_plugin_manager():
    pm = pluggy.PluginManager("pyccel")

    # Register exected hook format
    pm.add_hookspecs(hookspecs)

    # Search for available plugins on the system
    pm.load_setuptools_entrypoints("pyccel")

    # Register plugins provided inside Pyccel
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
        if not getattr(active_plugins, plugin_manager.get_name(plugin), False):
            plugin_manager.unregister(plugin)

