# pylint: disable=missing-function-docstring, missing-module-docstring

import os

import pytest

from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors import Errors
from pyccel.parser.parser import Parser
from pyccel.utilities.plugins import Plugins

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir, f) for f in files if (f.endswith(".py"))]


@pytest.mark.c
@pytest.mark.parametrize("f", files)
def test_codegen(f):
    plugins = Plugins()
    plugins.set_options({'accelerators': ['openmp']})
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f, output_folder=os.getcwd())
    ast = pyccel.parse(verbose=0)

    # Assert syntactic success
    assert not errors.has_errors()

    ast = pyccel.annotate(verbose=0)

    # Assert semantic success
    assert not errors.has_errors()

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, 'c', verbose=0)
    codegen.printer.doprint(codegen.ast)

    # Assert codegen success
    assert not errors.has_errors()
