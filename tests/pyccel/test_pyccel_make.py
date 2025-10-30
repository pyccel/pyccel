from pathlib import Path
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
    return request.param

#------------------------------------------------------------------------------
def pyccel_make_test(main_file, folder, language, build_system, args, output_dtype = float):
    python_output = get_python_output(folder / main_file, cwd = folder)

    p = subprocess.run(['pyccel-make', *args, f'--language={language}',
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
def test_project_abs_imports(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_abs_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
def test_project_class_imports(language, build_system):
    # Failing due to repeated renaming
    pyccel_make_test('runtest.py', current_folder / 'project_class_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Collisions are not handled"),
            pytest.mark.c]
        )
    )
)
def test_project_multi_imports(language, build_system):
    pyccel_make_test('file4.py', current_folder / 'project_multi_imports',
                     language, build_system, ['-f', 'file1.py', 'file2.py', 'file3.py', 'file4.py'],
                     output_dtype = str)

#------------------------------------------------------------------------------
def test_project_rel_imports(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_rel_imports',
                     language, build_system, ['-g', '**/*.py'])

#------------------------------------------------------------------------------
def test_project_containers(language, build_system):
    pyccel_make_test('runtest.py', current_folder / 'project_containers',
                     language, build_system, ['-d', str(current_folder / 'project_containers' / 'files.txt')],
                     output_dtype = int)
