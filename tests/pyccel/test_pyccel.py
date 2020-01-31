import subprocess
import os
import pytest
import shutil
import numpy as np

def get_abs_path(relative_path):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, relative_path)

def insert_pyccel_folder(abs_path):
    base_dir = os.path.dirname(abs_path)
    base_name = os.path.basename(abs_path)
    return os.path.join(base_dir, "__pyccel__", base_name)

def get_python_output(abs_path, cwd = None):
    if cwd is None:
        p = subprocess.Popen([shutil.which("python3") , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen([shutil.which("python3") , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def compile_pyccel(path_dir,test_file, options = ""):
    if options != "":
        options = options.split(' ')
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file] + options, universal_newlines=True, cwd=path_dir)
    else:
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def compile_fortran(path_dir,test_file,dependencies):
    root = insert_pyccel_folder(test_file)[:-3]

    assert(os.path.isfile(root+".f90"))

    command = [shutil.which("gfortran"), "-O3", "%s.f90" % root]
    if isinstance(dependencies, list):
        for d in dependencies:
            d = insert_pyccel_folder(d)
            command.append(d[:-3]+".o")
            command.append("-I"+os.path.dirname(d))
    elif isinstance(dependencies, str):
        dependencies = insert_pyccel_folder(dependencies)
        command.append(dependencies[:-3]+".o")
        command.append("-I"+os.path.dirname(dependencies))

    command.append("-o")
    command.append("%s" % test_file[:-3])

    p = subprocess.Popen(command, universal_newlines=True, cwd=path_dir)
    p.wait()

def get_fortran_output(abs_path):
    assert(os.path.isfile(abs_path))
    p = subprocess.Popen(["%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

def setup():
    teardown()

def teardown(path_dir = None):
    if path_dir is None:
        path_dir = os.path.dirname(os.path.realpath(__file__))
    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f == "__pyccel__":
            shutil.rmtree( file_name )
        elif not os.path.isfile(file_name):
            teardown(file_name)
        elif not f.endswith(".py"):
            os.remove(file_name)

def compare_pyth_fort_output( p_output, f_output ):
    p_output = p_output.strip().split()
    f_output = f_output.strip().split()

    assert(len(p_output) == len(f_output))
    for p, f in zip(p_output, f_output):
        p = float(p)
        f = float(f)
        assert(np.isclose(p,f))

def pyccel_test(test_file, dependencies = None, compile_with_pyccel = True, cwd = None, pyccel_commands = ""):
    if (cwd is None):
        cwd = os.path.dirname(test_file)

    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)
    pyth_out = get_python_output(test_file, cwd)
    if (isinstance(dependencies, list)):
        for d,i in enumerate(dependencies):
            dependencies[i] = get_abs_path(d)
            compile_pyccel(os.path.dirname(dependencies[i]), dependencies[i], pyccel_commands)
    elif (isinstance(dependencies, str)):
        dependencies = get_abs_path(dependencies)
        compile_pyccel(os.path.dirname(dependencies), dependencies, pyccel_commands)

    if compile_with_pyccel:
        compile_pyccel(cwd, test_file, pyccel_commands)
    else:
        compile_pyccel (cwd, test_file, pyccel_commands+"-t")
        compile_fortran(cwd, test_file, dependencies)

    fort_out = get_fortran_output(test_file[:-3])

    compare_pyth_fort_output(pyth_out, fort_out)

def test_rel_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_rel_imports import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/folder2_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/test_rel_imports.py"))

    p = subprocess.Popen([shutil.which("python3") , "%s" % base_dir+"/run_import_function.py", "scripts.folder2.test_rel_imports"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

def test_imports_compile():
    pyccel_test("scripts/test_imports.py","scripts/funcs.py", compile_with_pyccel = False)

def test_imports_in_folder():
    # Fails as imports are wrongly defined
    pyccel_test("scripts/test_folder_imports.py","scripts/folder1/folder1_funcs.py", compile_with_pyccel = False)

def test_imports():
    pyccel_test("scripts/test_imports.py","scripts/funcs.py")

def test_folder_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_imports2 import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir, "folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/test_imports2.py"))

    p = subprocess.Popen([shutil.which("python3") , "%s" % base_dir+"/run_import_function.py", "scripts.folder2.test_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

def test_folder_imports():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_imports2 import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir,"folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(os.path.join(path_dir,"folder2"), get_abs_path("scripts/folder2/test_imports2.py"))

    p = subprocess.Popen([shutil.which("python3") , "%s" % base_dir+"/run_import_function.py", "scripts.folder2.test_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

def test_funcs():
    pyccel_test("scripts/test_funcs.py")

def test_f2py_compat():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.test_f2py_compat import test_func

    pyth_out = str(test_func())

    compile_pyccel(path_dir, "test_f2py_compat.py")

    p = subprocess.Popen([shutil.which("python3") , "%s" % base_dir+"/run_import_function.py", "scripts.test_f2py_compat"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

def test_pyccel_calling_directory():
    cwd = get_abs_path(".")

    test_file = get_abs_path("scripts/test_funcs.py")
    pyth_out = get_python_output(test_file)

    compile_pyccel(cwd, test_file)

    fort_out = get_fortran_output(get_abs_path("scripts/test_funcs"))

    compare_pyth_fort_output( pyth_out, fort_out )

def test_in_specified():
    pyccel_test("scripts/test_degree_in.py")

@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/fib.py",
                                        "scripts/hope_benchmarks/pisum.py",
                                        "scripts/hope_benchmarks/ln_python.py",
                                        "scripts/hope_benchmarks/pairwise_python.py",
                                        "scripts/hope_benchmarks/point_spread_func.py",
                                        "scripts/hope_benchmarks/simplify.py",
                                        "scripts/hope_benchmarks_decorators/ln_python.py",
                                        "scripts/hope_benchmarks_decorators/pairwise_python.py",
                                        "scripts/hope_benchmarks_decorators/point_spread_func.py",
                                        "scripts/hope_benchmarks_decorators/simplify.py"
                                        ] )
def test_hope_benchmarks( test_file ):
    pyccel_test(test_file)

@pytest.mark.xfail
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks_decorators/fib.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",
                                        ] )
def test_hope_benchmarks_xfail( test_file ):
    pyccel_test(test_file)

@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import.py",
                                        "scripts/import_syntax/from_mod_import_as.py",
                                        "scripts/import_syntax/import_mod.py",
                                        "scripts/import_syntax/import_mod_as.py",
                                        "scripts/import_syntax/collisions.py"
                                        ] )
def test_import_syntax( test_file ):
    pyccel_test(test_file)
