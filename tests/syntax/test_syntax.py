# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import os

import pluggy
import pytest

from pyccel.errors.errors import Errors
from pyccel.naming import name_clash_checkers
from pyccel.parser.parser import Parser

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, "scripts")

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir, f) for f in files if (f.endswith(".py"))]


@pytest.mark.language_agnostic
@pytest.mark.parametrize("f", files)
def test_syntax(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()
    extensions = PluginManager()
    extensions.set_options({'openmp':True, 'omp_version':4.5})

    plugin_manager = pluggy.PluginManager("pyccel")

    pyccel = Parser(
        f,
        output_folder=os.getcwd(),
        name_clash_checker=name_clash_checkers["python"],
        plugin_manager=plugin_manager,
    )
    pyccel.parse(verbose=0)

    # Assert syntactic success
    assert not errors.has_errors()


######################
if __name__ == "__main__":
    print("*********************************")
    print("***                           ***")
    print("***      TESTING SYNTAX       ***")
    print("***                           ***")
    print("*********************************")

    for f in files:
        print("> testing {0}".format(str(os.path.basename(f))))
        test_syntax(f)
        print("\n")
