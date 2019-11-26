import subprocess
import os
import pytest
import shutil

def get_python_output(path_dir,test_file):
    p = subprocess.Popen([shutil.which("python3") , "%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def compile_pyccel(path_dir,test_file, options = ""):
    if options != "":
        p = subprocess.Popen([shutil.which("pyccel"), options, "%s" % test_file, "--include=."], universal_newlines=True, cwd=path_dir)
    else:
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file, "--include=."], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def compile_f2py(path_dir,test_file):
    root = test_file[:-3]
    p = subprocess.Popen([shutil.which("f2py"), "-c", "%s.f90" % root, "-m", "%s" % root], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def get_fortran_output(path_dir,test_file):
    p = subprocess.Popen(["%s/%s" % (path_dir , test_file)], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def pyccel_test(test_file, dependencies = None):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    pyth_out = get_python_output(path_dir,test_file)
    if (type(dependencies) is list):
        for d in dependencies:
            compile_pyccel(path_dir, d)
    elif (type(dependencies) is str):
        compile_pyccel(path_dir, dependencies)
    compile_pyccel(path_dir,test_file)
    fort_out = get_fortran_output(path_dir,test_file[:-3])

    assert(pyth_out.strip()==fort_out.strip())

@pytest.mark.xfail
def test_imports():
    pyccel_test("test_imports.py","funcs.py")

def test_funcs():
    pyccel_test("test_funcs.py")

def test_f2py_compat():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    from scripts.test_f2py_compat import return_one

    pyth_out = return_one()

    compile_pyccel(path_dir, "test_f2py_compat.py", "-f")
    compile_f2py(path_dir, "test_f2py_compat.py")

    import scripts.test_f2py_compat as mod
    fort_out = mod.return_one()

    assert(pyth_out==fort_out)
