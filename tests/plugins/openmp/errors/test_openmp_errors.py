# pylint: disable=missing-function-docstring, missing-module-docstring
import os

import pytest

from pyccel.plugins.plugin_tools import get_plugin_manager, handle_plugin_arguments
from pyccel.codegen.codegen import Codegen
from pyccel.codegen.pipeline import execute_pyccel
from pyccel.errors.errors import Errors, PyccelError
from pyccel.parser.parser import Parser


def get_files_from_folder(folder_name):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join(folder_name))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir, f) for f in files if (f.endswith(".py"))]
    return files


@pytest.mark.external
@pytest.mark.parametrize("f", get_files_from_folder("blockers"))
def test_blockers(f, language):
    plugin_manager = get_plugin_manager()
    handle_plugin_arguments(plugin_manager, {"openmp": True})

    errors = Errors()
    errors.reset()

    with pytest.raises(PyccelError):
        execute_pyccel(
                f,
                verbose=0,
                language=language,
                plugin_manager=plugin_manager,
            )
