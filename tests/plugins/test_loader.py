# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import pytest

from pyccel.errors.errors import Errors
from pyccel.parser.syntactic import SyntaxParser
from pyccel.utilities.plugins import Plugins

plugins = Plugins()

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')


@pytest.mark.external
def get_funcs(obj):
    methods = {}
    for name in dir(obj):
        if name.startswith('__'):
            continue

        attr = getattr(obj, name)
        if callable(attr) and hasattr(attr, '__func__'):
            methods[name] = attr.__func__
    return methods


@pytest.mark.external
def test_unload():
    plugins.unload_plugins()
    assert plugins.get_plugins() == []


@pytest.mark.external
def test_register():
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    plugins.load_plugins()
    parser = SyntaxParser(file)
    for plugin in plugins.get_plugins():
        assert plugin.is_registered(parser)


@pytest.mark.external
def test_openmp_register_unregister():
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    plugins.unload_plugins()
    parser_ref = SyntaxParser(file)

    plugins.load_plugins()
    omp_plugin = plugins.get_plugin('Openmp')
    plugins.set_plugins((omp_plugin,))
    plugins.set_options({'accelerators': ['openmp']})
    parser = SyntaxParser(file)

    modified_methods = get_funcs(parser)
    reference_methods = get_funcs(parser_ref)
    assert modified_methods != reference_methods

    plugins.unregister((parser,))
    modified_methods = get_funcs(parser)
    reference_methods = get_funcs(parser_ref)
    assert modified_methods == reference_methods


@pytest.mark.external
def test_openmp_register_refresh():
    errors = Errors()
    errors.reset()
    plugins.load_plugins()
    file = os.path.join(path_dir, 'omp5_specific.py')
    plugins.set_options({'accelerators': ['openmp']})
    parser = SyntaxParser(file)
    assert errors.has_warnings() == True
    errors.reset()

    # refresh is needed to patch with openmp 5.0
    plugins.set_options({'accelerators': ['openmp'], 'omp_version': 5.0})
    parser._syntax_done = False
    ast = parser.parse()
    assert errors.has_warnings() == True
    errors.reset()

    plugins.set_options({'accelerators': ['openmp'], 'omp_version': 5.0}, refresh=True)
    parser._syntax_done = False
    ast = parser.parse()
    assert errors.has_warnings() == False
    errors.reset()

    plugins.set_options({'accelerators': ['openmp'], 'omp_version': 4.5})
    parser._syntax_done = False
    ast = parser.parse()
    assert errors.has_warnings() == True
    errors.reset()

    # no refresh is needed since openmp 5.0 is already patched with
    plugins.set_options({'accelerators': ['openmp'], 'omp_version': 5.0})
    parser._syntax_done = False
    ast = parser.parse()
    assert errors.has_warnings() == False
    errors.reset()
