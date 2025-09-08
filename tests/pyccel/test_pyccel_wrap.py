# pylint: disable=missing-function-docstring, missing-module-docstring
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

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
    subprocess.run([compiler_info['exec'], '-shared', '-fPIC', '-o', output_folder / f'lib{stem}.so', input_folder / f'{stem}{low_level_suffix[language]}'],
                   check = True, cwd=cwd)

def check_pyccel_wrap_and_call_translation(low_level_stem, python_stem, language):
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
    """
    cwd = Path(__file__).parent / 'wrap_scripts' / f'{language}_tests'

    os.makedirs(cwd / '__pyccel__', exist_ok = True)

    python_file = cwd / f'{python_stem}.py'

    pyccel_flags = [f'--language={language}']
    if os.environ.get('PYCCEL_ERROR_MODE', 'user') == 'developer':
        pyccel_flags.append('--developer-mode')
        pyccel_flags.append('-vv')

    compile_low_level(low_level_stem, cwd, cwd, cwd / '__pyccel__', language)
    subprocess.run([shutil.which("pyccel-wrap"), cwd / f'{low_level_stem}.pyi', *pyccel_flags], check = True)
    py_run = subprocess.run([sys.executable, python_file], text = True, capture_output = True, cwd = cwd, check = True)
    subprocess.run([shutil.which("pyccel"), python_file, *pyccel_flags], check = True)
    
    exe_file = cwd / python_stem
    if sys.platform == "win32":
        exe_file = exe_file.with_suffix('.py')

    l_run = subprocess.run([exe_file], text = True, capture_output = True, cwd = cwd, check = True)

    return py_run.stdout, l_run.stdout


#--------------------------------------------------------------------------------------------------
#                                  Tests
#--------------------------------------------------------------------------------------------------
def test_function(language):
    check_pyccel_wrap_and_call_translation('functions', 'runtest_functions', language)

def test_class_accessors(language):
    check_pyccel_wrap_and_call_translation('class_property', 'runtest_class_property', language)
