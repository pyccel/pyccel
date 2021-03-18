# pylint: disable=missing-function-docstring, missing-module-docstring/
import os
import shutil
import pytest
from pyccel.commands.pyccel_clean import pyccel_clean

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c)
    ],
    scope='module'
)
def language(request):
    return request.param
#------------------------------------------------------------------------------

#==============================================================================
# PYTEST MODULE SETUP AND TEARDOWN
#==============================================================================

#------------------------------------------------------------------------------
def pytest_sessionfinish(session):
    path_dir = os.path.dirname(os.path.realpath(__file__))
    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

    config = session.config
    xdist_plugin = config.pluginmanager.getplugin("xdist")
    if xdist_plugin is None or "PYTEST_XDIST_WORKER_COUNT" not in os.environ \
            or os.getenv('PYTEST_XDIST_WORKER_COUNT') == 1:
        pyccel_clean(path_dir, remove_shared_libs = True)
