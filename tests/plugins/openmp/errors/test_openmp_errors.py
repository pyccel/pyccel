# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import pytest

from pyccel.errors.errors import Errors
from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.utilities.plugins import Plugins
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
    plugins = Plugins()
    plugins.set_options({'accelerators': ['openmp']})
    errors = Errors()
    errors.reset()

    with pytest.raises(PyccelError):
        pyccel = Parser(f)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        name = os.path.basename(f)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name, 'fortran')
        codegen.printer.doprint(codegen.ast)
