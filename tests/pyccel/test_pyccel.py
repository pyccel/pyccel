import subprocess
import os
import pytest
import shutil

def get_abs_path(relative_path):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, relative_path)

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
        p = subprocess.Popen([shutil.which("pyccel")] + options + ["%s" % test_file, "--include=."], universal_newlines=True, cwd=path_dir)
    else:
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file, "--include=."], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def compile_f2py(path_dir,test_file, dependencies = None):
    root = test_file[:-3]
    command = [shutil.which("f2py"), "-c", "%s.f90" % root]
    if isinstance(dependencies, list):
        for d in dependencies:
            command.append(d[:-3]+".o")
            command.append("-I"+os.path.dirname(d))
    elif isinstance(dependencies, str):
        command.append(dependencies[:-3]+".o")
        command.append("-I"+os.path.dirname(dependencies))

    command.append("-m")
    command.append("%s_call" % root)

    p = subprocess.Popen(command, universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

def compile_fortran(path_dir,test_file,dependencies):
    root = test_file[:-3]

    command = [shutil.which("gfortran"), "-O3", "%s.f90" % root]
    if isinstance(dependencies, list):
        for d in dependencies:
            command.append(d[:-3]+".o")
            command.append("-I"+os.path.dirname(d))
    elif isinstance(dependencies, str):
        command.append(dependencies[:-3]+".o")
        command.append("-I"+os.path.dirname(dependencies))

    command.append("-o")
    command.append("%s" % root)

    p = subprocess.Popen(command, universal_newlines=True, cwd=path_dir)
    p.wait()

def get_fortran_output(abs_path):
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
        if not os.path.isfile(file_name):
            teardown(file_name)
        elif not f.endswith(".py"):
            os.remove(file_name)

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

    assert(pyth_out.strip()==fort_out.strip())

def test_rel_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_rel_imports import testing

    pyth_out = testing()

    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/folder2_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/test_rel_imports.py"), "-f --output=folder2")
    p = subprocess.Popen([shutil.which("f2py"), "-c", "folder2_funcs.o", "test_rel_imports.f90", "-m", "test_rel_imports_call"],
            universal_newlines=True, cwd=os.path.join(path_dir,"folder2"))
    p.wait()
    assert(p.returncode==0)

    import scripts.folder2.test_rel_imports_call as mod
    fort_out = mod.test_rel_imports.testing()

    assert(pyth_out==fort_out)

def test_imports_compile():
    pyccel_test("scripts/test_imports.py","scripts/funcs.py", compile_with_pyccel = False)

def test_imports_in_folder():
    # Fails as imports are wrongly defined
    pyccel_test("scripts/test_folder_imports.py","scripts/folder1/folder1_funcs.py", compile_with_pyccel = False)

@pytest.mark.xfail
def test_imports():
    # Fails as pyccel cannot compile the resulting files
    pyccel_test("scripts/test_imports.py","scripts/funcs.py")

def test_folder_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_imports2 import testing

    pyth_out = testing()

    compile_pyccel(os.path.join(path_dir, "folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/test_imports2.py"), "-f")
    p = subprocess.Popen([shutil.which("f2py"), "-c", "../folder1/folder1_funcs.o", "../test_imports2.f90", "-m", "test_imports2_call", "-I../folder1"],
            universal_newlines=True, cwd=os.path.join(path_dir,"folder2"))
    p.wait()
    assert(p.returncode==0)

    import scripts.folder2.test_imports2_call as mod
    fort_out = mod.test_imports2.testing()

    assert(pyth_out==fort_out)

def test_folder_imports():
    # pyccel is called on scripts/folder2/test_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.test_imports2 import testing

    pyth_out = testing()

    compile_pyccel(os.path.join(path_dir,"folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(os.path.join(path_dir,"folder2"), get_abs_path("scripts/folder2/test_imports2.py"), "-f")
    compile_f2py(os.path.join(path_dir,"folder2"), "test_imports2.py", "../folder1/folder1_funcs.py")

    import scripts.folder2.test_imports2_call as mod
    fort_out = mod.test_imports2.testing()

    assert(pyth_out==fort_out)

def test_funcs():
    pyccel_test("scripts/test_funcs.py")

def test_f2py_compat():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.test_f2py_compat import return_one

    pyth_out = return_one()

    compile_pyccel(path_dir, "test_f2py_compat.py", "-f")
    compile_f2py(path_dir, "test_f2py_compat.py")

    import scripts.test_f2py_compat_call as mod
    fort_out = mod.test_f2py_compat.return_one()

    assert(pyth_out==fort_out)

def test_pyccel_calling_directory():
    cwd = get_abs_path(".")

    test_file = get_abs_path("scripts/test_funcs.py")
    pyth_out = get_python_output(test_file)

    compile_pyccel(cwd, test_file)

    fort_out = get_fortran_output(get_abs_path("test_funcs"))

    assert(pyth_out.strip()==fort_out.strip())

def test_in_specified():
    pyccel_test("scripts/test_degree_in.py")
