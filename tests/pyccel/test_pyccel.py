import subprocess
import os
import pytest
import shutil

def get_python_output(path_dir,test_file):
    p = subprocess.Popen([shutil.which("python3") , "%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def compile_pyccel(path_dir,test_file):
    p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file, "--include=."], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def get_fortran_output(path_dir,test_file):
    p = subprocess.Popen(["%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def pyccel_test(test_file, dependencies):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    pyth_out = get_python_output(path_dir,test_file)
    for d in dependencies:
        compile_pyccel(path_dir, d)
    compile_pyccel(path_dir,test_file)
    fort_out = get_fortran_output(path_dir,test_file[:-3])

    assert(pyth_out.strip()==fort_out.strip())

@pytest.mark.xfail
def test_imports():
    pyccel_test("test_imports.py",["funcs.py"])

def test_funcs():
    pyccel_test("test_funcs.py")
