# pylint: disable=missing-function-docstring, missing-module-docstring
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_pyccel import compare_pyth_fort_output
from pyccel.compilers.default_compilers import available_compilers

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    ],
    scope = "session"
)
def language(request):
    return request.param

#--------------------------------------------------------------------------------------------------
#                              Utility functions
#--------------------------------------------------------------------------------------------------

low_level_suffix = {'c': '.c',
                    'fortran': '.f90'}

def compile_low_level(stem, input_folder, output_folder, cwd, language):
    """
    Compile a low-level file to a library.

    Parameters
    ----------
    stem : str
        The stem of the low-level file being compiled.
    input_folder : str
        The folder containing the low-level file being compiled.
    output_folder : str
        The folder where the library should be outputted.
    cwd : str
        The folder where the command will be run from.
    language : str
        The language we are compiling from.
    """
    compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    compiler_info = available_compilers[compiler_family][language]
    lib_suffix = '.dll' if sys.platform == 'win32' else '.so'
    subprocess.run([compiler_info['exec'], '-shared', '-fPIC', '-o',
                    output_folder / f'lib{stem}{lib_suffix}',
                    input_folder / f'{stem}{low_level_suffix[language]}'],
                   check = True, cwd=cwd)

def check_pyccel_wrap_and_call_translation(low_level_stem, python_stem, language, extra_flags = ()):
    """
    Check that pyccel-wrap allows a Python file to call a low-level file and that
    the Python code which calls that low-level file can itself be translated.

    Parameters
    ----------
    low_level_stem : str
        The stem of the low-level file containing the code we would like to call.
    python_stem : str
        The stem of the Python file which calls the low-level code.
    language : str
        The language we are compiling from.
    extra_flags : Iterable[str]
        Any extra flags to be passed to the wrap command.

    Results
    -------
    py_run_output : str
        The output of running the Python file which calls the code from the
        wrapped low-level file.
    l_run_output : str
        The output of running the low-level file obtained by translating
        the Python file which calls the low-level code.
    """
    cwd = Path(__file__).parent / 'wrap_scripts' / f'{language}_tests'

    pyccel_dirname = '__pyccel__mod__'
    os.makedirs(cwd / pyccel_dirname, exist_ok = True)

    python_file = cwd / f'{python_stem}.py'

    pyccel_flags = [f'--language={language}', *extra_flags]
    if os.environ.get('PYCCEL_ERROR_MODE', 'user') == 'developer':
        pyccel_flags.append('--developer-mode')
        pyccel_flags.append('-vv')

    compile_low_level(low_level_stem, cwd, cwd, cwd / pyccel_dirname, language)
    subprocess.run([shutil.which("pyccel-wrap"), cwd / f'{low_level_stem}.pyi', *pyccel_flags], check = True)
    py_run = subprocess.run([sys.executable, python_file], text = True, capture_output = True, cwd = cwd, check = True)
    subprocess.run([shutil.which("pyccel"), python_file, *pyccel_flags], check = True)

    exe_file = cwd / python_stem
    if sys.platform == "win32":
        exe_file = exe_file.with_suffix('.exe')

    l_run = subprocess.run([exe_file], text = True, capture_output = True, cwd = cwd, check = True)

    return py_run.stdout, l_run.stdout


#--------------------------------------------------------------------------------------------------
#                                  Tests
#--------------------------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_function(language):
    py_out, l_out = check_pyccel_wrap_and_call_translation('functions', 'runtest_functions', language)
    compare_pyth_fort_output(py_out, l_out, int, language)

def test_class_accessors(language):
    py_out, l_out = check_pyccel_wrap_and_call_translation('class_property', 'runtest_class_property', language)
    compare_pyth_fort_output(py_out, l_out, int, language)

def test_class_renaming(language):
    py_out, l_out = check_pyccel_wrap_and_call_translation('class_renaming', 'runtest_class_renaming', language)
    compare_pyth_fort_output(py_out, l_out, int, language)

def test_array_methods():
    # C is not tested as compiling array dependencies by hand is harder than necessary for the test
    py_out, l_out = check_pyccel_wrap_and_call_translation('array_methods', 'runtest_array_methods', 'fortran')
    compare_pyth_fort_output(py_out, l_out, float, language)

def test_overload_methods(language):
    py_out, l_out = check_pyccel_wrap_and_call_translation('class_overloaded_methods', 'runtest_class_overloaded_methods', language)
    compare_pyth_fort_output(py_out, l_out, [int, float], language)

def test_class_no_init(language):
    py_out, l_out = check_pyccel_wrap_and_call_translation('class_no_init', 'runtest_class_no_init', language)
    compare_pyth_fort_output(py_out, l_out, int, language)

def test_class_final_fortran_keyword():
    py_out, _ = check_pyccel_wrap_and_call_translation('final_destroy', 'runtest_final_destroy', 'fortran')
    assert "c allocated, cleaning up" in py_out

    # Variables declared in the main program implicitly acquire the save attribute according to the standard. The compiler is not supposed to generate any automatic finalization.
    #assert "c allocated, cleaning up" in l_out

    valgrind = shutil.which('valgrind')
    if valgrind:
        cwd = Path(__file__).parent / 'wrap_scripts' / 'fortran_tests'
        subprocess.run([valgrind, sys.executable, cwd / 'runtest_final_destroy.py'], cwd = cwd, check = True)
        exe_file = cwd / 'runtest_final_destroy'
        if sys.platform == "win32":
            exe_file = exe_file.with_suffix('.exe')
        subprocess.run([valgrind, exe_file], cwd = cwd, check = True)

# Flag tests

@pytest.mark.xdist_incompatible
@pytest.mark.parametrize('extra_flag', ['--mpi', '--openmp', '--time-execution', '--verbose', '--developer-mode'])
def test_accelerator_flags(language, extra_flag):
    py_out, l_out = check_pyccel_wrap_and_call_translation('functions', 'runtest_functions', language, (extra_flag,))
    compare_pyth_fort_output(py_out, l_out, int, language)

@pytest.mark.xdist_incompatible
def test_convert_only(language):
    cwd = Path(__file__).parent / 'wrap_scripts' / f'{language}_tests'

    pyccel_dirname = '__pyccel__' + os.environ.get('PYTEST_XDIST_WORKER', '')
    pyccel_mod_dirname = '__pyccel__mod__'
    os.makedirs(cwd / pyccel_mod_dirname, exist_ok = True)

    pyccel_flags = ['--convert-only', '--time-execution']
    if os.environ.get('PYCCEL_ERROR_MODE', 'user') == 'developer':
        pyccel_flags.append('--developer-mode')
        pyccel_flags.append('-vv')

    compile_low_level('functions', cwd, cwd, cwd / pyccel_mod_dirname, language)
    p = subprocess.run([shutil.which("pyccel-wrap"), cwd / 'functions.pyi', *pyccel_flags], check = True, text = True, capture_output = True)
    assert 'Time' in p.stdout
    assert (cwd / pyccel_dirname / 'functions_wrapper.c').exists()
