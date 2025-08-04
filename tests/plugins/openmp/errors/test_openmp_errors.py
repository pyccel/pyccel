# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import pytest

from pyccel.errors.errors import Errors
from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.utilities.pluginmanager import PluginManager
from pyccel.errors.errors import PyccelError

def get_files_from_folder(folder_name):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join(folder_name))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files


@pytest.mark.external
@pytest.mark.parametrize("f",get_files_from_folder("blockers"))
def test_blockers(f):
    plugins = PluginManager()
    plugins.set_options({'openmp':True})
    errors = Errors()
    errors.reset()

    with pytest.raises(PyccelError):
        pyccel = Parser(f, output_folder=os.getcwd())
        ast = pyccel.parse(verbose=0)

        ast = pyccel.annotate(verbose=0)

        name = os.path.basename(f)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name, 'fortran', verbose=0)
        codegen.printer.doprint(codegen.ast)
