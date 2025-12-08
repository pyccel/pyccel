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
    Test the pyccel make command.

    Parameters
    ----------
    main_file : str
        The file (relative to folder) containing the `__main__` implementation.
    folder : Path
        The absolute path to the folder in which the test is found.
    language : {'c', 'fortran', 'python'}
        The language used for translation.
    build_system : {'cmake', 'meson'}
        The build system that should be used for compilation.
    args : iterable[str]
        The additional arguments that should be passed to pyccel make. This must
        include a command to specify how the files to be translated are selected.
    output_dtype : type/list of types, default=float
        The types expected as output of the program.
        If one argument is provided then all types are assumed to be the same.
    """
    python_output = get_python_output(folder / main_file, cwd = folder)

    p = subprocess.run([shutil.which('pyccel'), 'make', *args, f'--language={language}',
                        f'--build-system={build_system}'], cwd=folder, check=True)

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

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.parametrize('extra_flag', ['--mpi', '--openmp', '--time-execution', '--verbose', '--developer-mode', '-vv'])
def test_flags(language, build_system, extra_flag):
    if extra_flag == '--mpi' and sys.platform == "win32":
        return pytest.skip(reason="Meson does not correctly handle spaces in paths")

    pyccel_make_test('file4.py', current_folder / 'project_multi_imports',
                     language, build_system, ['-f', 'file1.py', 'file2.py', 'file3.py', 'file4.py', extra_flag],
                     output_dtype = str)

    return None

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_output_flag(language, build_system):
    main_file = 'file4.py'
    folder = current_folder / 'project_multi_imports'
    args = ['-f', 'file1.py', 'file2.py', 'file3.py', 'file4.py', '--output', 'outfolder']

    python_output = get_python_output(folder / main_file, cwd = folder)

    p = subprocess.run([shutil.which('pyccel'), 'make', *args, f'--language={language}',
                        f'--build-system={build_system}'], cwd=folder, check=True)

    exe_path = (folder / 'outfolder' / main_file).with_suffix('')

    if language == "python":
        lang_output = get_python_output(exe_path.with_suffix('.py'))
    else:
        if sys.platform == "win32":
            exe_path = exe_path.with_suffix('.exe')
        p = subprocess.run([exe_path], capture_output = True, text=True, check=True)
        lang_output = p.stdout

    compare_pyth_fort_output(python_output, lang_output, str, language)

    # Clean up after test
    shutil.rmtree(folder / 'outfolder', ignore_errors=True)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_circular_dependencies(language, build_system):
    p = subprocess.run([shutil.which('pyccel'), 'make', '-f', 'runtest.py', 'src/folder1/file1.py',
                        'src/folder2/file2.py', 'src/folder1/file3.py', f'--language={language}',
                        f'--build-system={build_system}'], cwd=current_folder / 'project_circular_imports',
                       check=False, capture_output=True, text=True)

    assert p.returncode != 0
    assert "Found circular dependencies between directories" in p.stdout
