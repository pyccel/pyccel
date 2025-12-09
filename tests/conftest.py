# pylint: disable=missing-function-docstring, missing-module-docstring
import logging
import os
import shutil
import pytest
from mpi4py import MPI
from pyccel.commands.pyccel_clean import pyccel_clean

github_debugging = 'DEBUG' in os.environ
if github_debugging:
    import sys
    sys.stdout = sys.stderr

language_markers = ('language_agnostic', 'fortran', 'c', 'python')

@pytest.fixture(autouse=True)
def check_language_markers(request):
    """
    Ensure all tests have a language marker.
    """
    markers = [m.name for m in request.node.iter_markers()]
    assert len(markers) > 0
    assert any(l in markers for l in language_markers)

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "session"
)
def language(request):
    return request.param

@pytest.fixture( params=[
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ],
    scope = "session"
)
def stc_language(request):
    return request.param

@pytest.fixture(autouse=True)
def skipif_by_language(request):
    """
    Looks for the decorator `skipif_by_language(condition, *, language, reason)`
    and skips the test when the condition is met for a given language.

    """
    mark = request.node.get_closest_marker('skipif_by_language')
    if mark:

        preamble = "Error evaluating 'skipif_by_language': "
        assert len(mark.args) == 1, preamble + "you need to specify a boolean condition"
        assert 'language' in mark.kwargs.keys(), preamble + "you need to specify language=STRING"
        assert 'reason'   in mark.kwargs.keys(), preamble + "you need to specify reason=STRING"

        condition = mark.args[0]
        language  = mark.kwargs['language']
        reason    = mark.kwargs['reason']

        assert isinstance(condition, bool), preamble + "condition must be bool"
        assert isinstance(language, str), preamble + "language must be str"
        assert isinstance(reason, str), preamble + "reason must be str"

        assert 'language' in request.fixturenames, preamble + "test must depend on the language fixture"

        if condition and request.getfixturevalue('language') == language:
            pytest.skip(reason)

def move_coverage(path_dir):
    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

def pytest_runtest_teardown(item, nextitem):
    path_dir = os.path.dirname(os.path.realpath(item.fspath))
    move_coverage(path_dir)

    config = item.config
    xdist_plugin = config.pluginmanager.getplugin("xdist")
    if xdist_plugin is None or "PYTEST_XDIST_WORKER_COUNT" not in os.environ \
            or os.getenv('PYTEST_XDIST_WORKER_COUNT') == 1:
        marks = [m.name for m in item.own_markers ]
        if 'mpi' not in marks:
            pyccel_clean(path_dir, remove_shared_libs = True)
        else:
            comm = MPI.COMM_WORLD
            comm.Barrier()
            if comm.rank == 0:
                pyccel_clean(path_dir, remove_shared_libs = True)
            comm.Barrier()

def pytest_addoption(parser):
    parser.addoption("--developer-mode", action="store_true", default=github_debugging, help="Show tracebacks when pyccel errors are raised")

def pytest_sessionstart(session):
    # setup_stuff
    if session.config.option.developer_mode:
        os.environ['PYCCEL_ERROR_MODE'] = 'developer'

    if github_debugging:
        logging.basicConfig()
        logging.getLogger("filelock").setLevel(logging.DEBUG)

    # Clean path before beginning but never delete anything in parallel mode
    path_dir = os.path.dirname(os.path.realpath(__file__))

    config = session.config
    xdist_plugin = config.pluginmanager.getplugin("xdist")
    if xdist_plugin is None:
        marks = [m.name for m in session.own_markers ]
        if 'mpi' not in marks:
            pyccel_clean(path_dir)

def pytest_runtest_setup(item):
    # Skip on `skip_llvm` marker and environment variable
    if 'skip_llvm' in item.keywords:
        if os.environ.get('PYCCEL_DEFAULT_COMPILER', '').lower() == 'llvm':
            pytest.skip("Skipping test because PYCCEL_DEFAULT_COMPILER=LLVM")

