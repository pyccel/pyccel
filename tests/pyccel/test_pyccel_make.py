# pylint: disable=missing-function-docstring, missing-module-docstring
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from test_pyccel import get_python_output, compare_pyth_fort_output

current_folder = Path(__file__).parent

@pytest.fixture( params=[
        pytest.param("cmake", marks = pytest.mark.cmake),
        pytest.param("meson", marks = pytest.mark.meson),
    ],
)
def build_system(request):
    """
    Test with the chosen build system.
    """
    return request.param

#------------------------------------------------------------------------------
def pyccel_make_test(main_file, folder, language, build_system, args, output_dtype = float):
    """
    Test the pyccel-make command.

    Parameters
    ----------
    main_file : str
        The file containing the `__main__` implementation.
    folder : Path
        The folder in which the test is found.
    build_system : str
        The build system that should be used for compilation.
    args : iterable[str]
        The additional arguments that should be passed to pyccel-make. This must
        include a command to specify how the files to be translated are selected.
    output_dtype : type/list of types, default=float
        The types expected as output of the program.
        If one argument is provided then all types are assumed to be the same.
    """
    python_output = get_python_output(folder / main_file, cwd = folder)

    p = subprocess.run([shutil.which('pyccel-make'), *args, f'--language={language}',
                        '-vv', f'--build-system={build_system}'], cwd=folder, check=True)

    exe_path = (folder / main_file).with_suffix('')

    if language == "python":
        lang_output = get_python_output(exe_path.with_suffix('.py'))
    else:
        if sys.platform == "win32":
            exe_path = exe_path.with_suffix('.exe')
        p = subprocess.run([exe_path], capture_output = True, text=True, check=True)
        lang_output = p.stdout

    compare_pyth_fort_output(python_output, lang_output, output_dtype, language)

    # Check if main can still run via Python
    if language != "python":
        p = subprocess.run([sys.executable, exe_path.with_suffix('.py')], capture_output = True,
                           text=True, check=True)
        assert p.stdout == python_output

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_project_abs_imports(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_abs_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_project_class_imports(language, build_system):
    # Failing due to repeated renaming
    pyccel_make_test('runtest.py', current_folder / 'project_class_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_project_multi_imports(language, build_system):
    pyccel_make_test('file4.py', current_folder / 'project_multi_imports',
                     language, build_system, ['-f', 'file1.py', 'file2.py', 'file3.py', 'file4.py'],
                     output_dtype = str)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_project_rel_imports(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_rel_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_project_containers(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_containers',
                     language, build_system, ['-d', str(current_folder / 'project_containers' / 'files.txt')],
                     output_dtype = int)
