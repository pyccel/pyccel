# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring, protected-access
import os
import inspect
from unittest.mock import patch

import pytest

from pyccel.errors.errors import Errors
from pyccel.errors.messages import OMP_VERSION_NOT_SUPPORTED, PLUGIN_DIRECTORY_NOT_FOUND
from pyccel.parser.syntactic import SyntaxParser
from pyccel.plugins.Openmp.plugin import Openmp
from pyccel.utilities.pluginmanager import PluginManager

errors = Errors()
errors.reset()
plugins = PluginManager()

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')


@pytest.mark.external
def get_funcs(obj):
    methods = {}
    for name in dir(obj):
        if name.startswith('__'):
            continue
        if inspect.isdatadescriptor(getattr(type(obj), name, None)):
            continue
        attr = getattr(obj, name)
        if callable(attr) and hasattr(attr, '__func__'):
            methods[name] = attr.__func__
    return methods


@pytest.mark.external
@patch('pyccel.plugins.Openmp.plugin.errors.report')
@patch('pyccel.utilities.pluginmanager.os.path.isdir')
def test_load(mock_isdir, mock_report):
    errors.reset()
    mock_isdir.return_value = False
    plugins.load_plugins('some_dir')
    mock_report.assert_called_with(
        PLUGIN_DIRECTORY_NOT_FOUND.format('some_dir'),
        severity='warning')


@pytest.mark.external
def test_unload():
    plugins.unload_plugins()
    assert plugins.get_plugins() == []


@pytest.mark.external
def test_register():
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    plugins.load_plugins()
    parser = SyntaxParser(file, verbose=0)
    for plugin in plugins.get_plugins():
        assert plugin.is_registered(parser)


@pytest.mark.external
@patch('pyccel.plugins.Openmp.plugin.errors.report')
def test_openmp_resolve_version(mock_report):
    errors.reset()
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    ver = 5.1
    plugins.set_options({'openmp': True, 'omp_version': ver})
    SyntaxParser(file, verbose=0)
    mock_report.assert_called_with(
        OMP_VERSION_NOT_SUPPORTED.format(ver, Openmp.DEFAULT_VERSION),
        severity='warning')


@pytest.mark.external
def test_openmp_no_implementation():
    class NoImp:
        def no_imp_method(self):
            pass

    plugins.load_plugins()
    plugins.set_options({'openmp': True})
    ins = NoImp()
    reference_methods = get_funcs(ins)
    plugins.register((ins,))
    modified_methods = get_funcs(ins)
    assert reference_methods == modified_methods


@pytest.mark.external
def test_openmp_register_deregister():
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    plugins.unload_plugins()
    parser_ref = SyntaxParser(file, verbose=0)

    plugins.load_plugins()
    omp_plugin = plugins.get_plugin('Openmp')
    plugins.set_plugins((omp_plugin,))
    plugins.set_options({'openmp': True})
    parser = SyntaxParser(file, verbose=0)

    modified_methods = get_funcs(parser)
    reference_methods = get_funcs(parser_ref)
    assert modified_methods != reference_methods

    plugins.deregister((parser,))
    modified_methods = get_funcs(parser)
    reference_methods = get_funcs(parser_ref)
    assert modified_methods == reference_methods


@pytest.mark.external
def test_openmp_register_refresh():
    plugins.load_plugins()
    file = os.path.join(path_dir, 'omp5_specific.py')
    plugins.set_options({'openmp': True})
    parser = SyntaxParser(file, verbose=0)
    assert errors.has_warnings()
    errors.reset()

    # refresh is needed to patch with openmp 5.0
    plugins.set_options({'openmp': True, 'omp_version': 5.0})
    parser._syntax_done = False
    parser.parse()
    assert errors.has_warnings()
    errors.reset()

    plugins.set_options({'openmp': True, 'omp_version': 5.0}, refresh=True)
    parser._syntax_done = False
    parser.parse()
    assert not errors.has_warnings()
    errors.reset()

    plugins.set_options({'openmp': True, 'omp_version': 4.5})
    parser._syntax_done = False
    parser.parse()
    assert errors.has_warnings()
    errors.reset()

    # no refresh is needed since openmp 5.0 is already patched with
    plugins.set_options({'openmp': True, 'omp_version': 5.0})
    parser._syntax_done = False
    parser.parse()
    assert not errors.has_warnings()
    errors.reset()


@pytest.mark.external
def test_openmp_same_version_refresh():
    plugins.unload_plugins()
    plugins.load_plugins()
    file = os.path.join(path_dir, 'any_omp4_specific.py')
    plugins.set_options({'openmp': True})
    parser = SyntaxParser(file, verbose=0)
    parser.parse()
    with patch.object(plugins.get_plugin('Openmp'), '_apply_patches') as mock_apply_patches:
        plugins.set_options({'openmp': True}, refresh=True)
        assert mock_apply_patches.call_count == 0
