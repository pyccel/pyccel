import subprocess
import os
import shutil
import numpy as np
import pytest

def get_python_output(path_dir,test_file):
    p = subprocess.Popen(["python3" , "%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, error = p.communicate()
    assert(p.returncode==0)
    return out

def compile_pyccel(path_dir,test_file):
    p = subprocess.Popen(["pyccel", "%s" % test_file], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def get_fortran_output(path_dir,test_file):
    p = subprocess.Popen(["%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, error = p.communicate()
    assert(p.returncode==0)
    return out

def pyccel_test(test_file):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    pyth_out = get_python_output(path_dir,test_file)
    compile_pyccel(path_dir,test_file)
    fort_out = get_fortran_output(path_dir,test_file[:-3])

    assert(pyth_out.strip()==fort_out.strip())

@pytest.mark.xfail
def test_imports():
    pyccel_test("test_imports.py")

def test_funcs():
    pyccel_test("test_funcs.py")
