import subprocess
import os
import pytest
import shutil
import numpy as np
import re
import sys

#==============================================================================
# UTILITIES
#==============================================================================

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Lack of print support"),
            pytest.mark.c]
        )
    ],
    scope='module'
)
def language(request):
    return request.param
#------------------------------------------------------------------------------

def get_abs_path(relative_path):
    relative_path = os.path.normpath(relative_path)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, relative_path)

#------------------------------------------------------------------------------
def get_exe(filename):
    exefile = os.path.splitext(filename)[0]
    if sys.platform == "win32":
        exefile = exefile + ".exe"
    return exefile

#------------------------------------------------------------------------------
def insert_pyccel_folder(abs_path):
    base_dir = os.path.dirname(abs_path)
    base_name = os.path.basename(abs_path)
    return os.path.join(base_dir, "__pyccel__", base_name)

#------------------------------------------------------------------------------
def get_python_output(abs_path, cwd = None):
    if cwd is None:
        p = subprocess.Popen([sys.executable , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen([sys.executable , "%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

#------------------------------------------------------------------------------
def compile_pyccel(path_dir,test_file, options = ""):
    if options != "":
        options = options.strip().split(' ')
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file] + options, universal_newlines=True, cwd=path_dir)
    else:
        p = subprocess.Popen([shutil.which("pyccel"), "%s" % test_file], universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
def get_fortran_output(abs_path):
    assert(os.path.isfile(abs_path))
    p = subprocess.Popen(["%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
    out, _ = p.communicate()
    assert(p.returncode==0)
    return out

#------------------------------------------------------------------------------
def get_value(string, regex, conversion):
    match = regex.search(string)
    assert(match)
    value = conversion(match.group())
    string = string[match.span()[1]:]
    return value, string

def compare_pyth_fort_output_by_type( p_output, f_output, dtype=float ):

    if dtype is str:
        assert(p_output.strip()==f_output.strip())
    elif dtype is complex:
        rx = re.compile('[-0-9.eEj]+')
        p, p_output = get_value(p_output, rx, complex)
        if p.imag == 0:
            p2, p_output = get_value(p_output, rx, complex)
            p = p+p2

        rx = re.compile('[-0-9.eE]+')
        f, f_output  = get_value(f_output, rx, float)
        f2, f_output = get_value(f_output, rx, float)
        f = f+f2*1j
        assert(np.isclose(p,f))
    elif dtype is bool:
        rx = re.compile('TRUE|True|true|1|T|t|FALSE|False|false|F|f|0')
        bool_conversion = lambda m: m.lower() in ['true', 't', '1']
        p, p_output = get_value(p_output, rx, bool_conversion)
        f, f_output = get_value(f_output, rx, bool_conversion)
        assert(p==f)

    elif dtype is float:
        rx = re.compile('[-0-9.eE]+')
        p, p_output = get_value(p_output, rx, float)
        f, f_output = get_value(f_output, rx, float)
        assert(np.isclose(p,f))

    elif dtype is int:
        rx = re.compile('[-0-9eE]+')
        p, p_output = get_value(p_output, rx, int)
        f, f_output = get_value(f_output, rx, int)
        assert(p==f)
    else:
        raise NotImplementedError("Type comparison not implemented")
    return p_output,f_output

#------------------------------------------------------------------------------
def compare_pyth_fort_output( p_output, f_output, dtype=float ):

    if isinstance(dtype,list):
        for d in dtype:
            p_output,f_output = compare_pyth_fort_output_by_type(p_output,f_output,d)
    elif dtype is complex:
        while len(p_output)>0 and len(f_output)>0:
            p_output,f_output = compare_pyth_fort_output_by_type(p_output,f_output,complex)
    else:
        p_output = p_output.strip().split()
        f_output = f_output.strip().split()
        for p, f in zip(p_output, f_output):
            compare_pyth_fort_output_by_type(p,f,dtype)

#------------------------------------------------------------------------------
def pyccel_test(test_file, dependencies = None, compile_with_pyccel = True,
        cwd = None, pyccel_commands = "", output_dtype = float,
        language = None):
    test_file = os.path.normpath(test_file)

    if (cwd is None):
        cwd = os.path.dirname(test_file)

    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)
    pyth_out = get_python_output(test_file, cwd)
    if (isinstance(dependencies, list)):
        for i, d in enumerate(dependencies):
            dependencies[i] = get_abs_path(d)
            compile_pyccel(os.path.dirname(dependencies[i]), dependencies[i], pyccel_commands)
    elif (isinstance(dependencies, str)):
        dependencies = get_abs_path(dependencies)
        compile_pyccel(os.path.dirname(dependencies), dependencies, pyccel_commands)

    if language:
        pyccel_commands += " --language="+language
    if compile_with_pyccel:
        compile_pyccel(cwd, test_file, pyccel_commands)
    else:
        compile_pyccel (cwd, test_file, pyccel_commands+"-t")
        compile_fortran(cwd, test_file, dependencies)

    fort_out = get_fortran_output(get_exe(test_file))

    compare_pyth_fort_output(pyth_out, fort_out, output_dtype)

#==============================================================================
# PYTEST MODULE SETUP AND TEARDOWN
#==============================================================================
def setup():
    teardown()

#------------------------------------------------------------------------------
def teardown(path_dir = None):
    if path_dir is None:
        path_dir = os.path.dirname(os.path.realpath(__file__))

    for root, _, files in os.walk(path_dir):
        for name in files:
            if name.startswith(".coverage"):
                shutil.copyfile(os.path.join(root,name),os.path.join(os.getcwd(),name))

    files = os.listdir(path_dir)
    for f in files:
        file_name = os.path.join(path_dir,f)
        if f == "__pyccel__":
            shutil.rmtree( file_name )
        elif not os.path.isfile(file_name):
            teardown(file_name)
        elif not f.endswith(".py"):
            os.remove(file_name)

#==============================================================================
# UNIT TESTS
#==============================================================================
def test_relative_imports_in_project():

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_rel_imports")
    pyth_out = get_python_output('runtest.py', cwd=path_dir)

    compile_pyccel(path_dir, 'project/folder1/mod1.py')
    compile_pyccel(path_dir, 'project/folder2/mod2.py')
    compile_pyccel(path_dir, 'project/folder2/mod3.py')
    fort_out = get_python_output('runtest.py', cwd=path_dir)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_absolute_imports_in_project():

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_abs_imports")
    pyth_out = get_python_output('runtest.py', cwd=path_dir)

    compile_pyccel(path_dir, 'project/folder1/mod1.py')
    compile_pyccel(path_dir, 'project/folder2/mod2.py')
    compile_pyccel(path_dir, 'project/folder2/mod3.py')
    fort_out = get_python_output('runtest.py', cwd=path_dir)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_rel_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/runtest_rel_imports.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_rel_imports import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/folder2_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/runtest_rel_imports.py"))

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_rel_imports"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_imports_compile():
    pyccel_test("scripts/runtest_imports.py","scripts/funcs.py", compile_with_pyccel = False)

#------------------------------------------------------------------------------
def test_imports_in_folder():
    # Fails as imports are wrongly defined
    pyccel_test("scripts/runtest_folder_imports.py","scripts/folder1/folder1_funcs.py", compile_with_pyccel = False)

#------------------------------------------------------------------------------
def test_imports():
    pyccel_test("scripts/runtest_imports.py","scripts/funcs.py")

#------------------------------------------------------------------------------
def test_folder_imports_python_accessible_folder():
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_imports2 import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir, "folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/runtest_imports2.py"))

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_folder_imports():
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_imports2 import test_func

    pyth_out = str(test_func())

    compile_pyccel(os.path.join(path_dir,"folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"))
    compile_pyccel(os.path.join(path_dir,"folder2"), get_abs_path("scripts/folder2/runtest_imports2.py"))

    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), "scripts.folder2.runtest_imports2"],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
def test_funcs(language):
    pyccel_test("scripts/runtest_funcs.py", language = language)

#------------------------------------------------------------------------------
def test_bool():
    pyccel_test("scripts/bool_comp.py", output_dtype = bool)

#------------------------------------------------------------------------------
def test_expressions():
    types = [float, complex, int, float, float, int] + [float]*3 + \
            [complex, int, complex, complex, int, float] + [complex]*3 + \
            [float]*3 + [int] + [float]*2 + [int] + [float]*3 + [int] + \
            [float]*3 + [int]*2 + [float]*2 + [int]*5 + [complex]
    pyccel_test("scripts/expressions.py",
                output_dtype = types)

#------------------------------------------------------------------------------
def test_default_arguments():
    pyccel_test("scripts/runtest_default_args.py",
            dependencies = "scripts/default_args_mod.py",
            output_dtype = [int,int,float,float,float,
                float,float,float,float,bool,bool,bool,
                float,float,float,float])

#------------------------------------------------------------------------------
def test_pyccel_calling_directory():
    cwd = get_abs_path(".")

    test_file = get_abs_path("scripts/runtest_funcs.py")
    pyth_out = get_python_output(test_file)

    compile_pyccel(cwd, test_file)

    fort_out = get_fortran_output(get_exe(test_file))

    compare_pyth_fort_output( pyth_out, fort_out )

#------------------------------------------------------------------------------
def test_in_specified():
    pyccel_test("scripts/runtest_degree_in.py")

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/pisum.py",
                                        "scripts/hope_benchmarks/ln_python.py",
                                        "scripts/hope_benchmarks/pairwise_python.py",
                                        "scripts/hope_benchmarks/point_spread_func.py",
                                        "scripts/hope_benchmarks/simplify.py",
                                        "scripts/hope_benchmarks_decorators/ln_python.py",
                                        "scripts/hope_benchmarks_decorators/pairwise_python.py",
                                        "scripts/hope_benchmarks_decorators/point_spread_func.py",
                                        "scripts/hope_benchmarks_decorators/simplify.py",
                                        "scripts/hope_benchmarks_decorators/fib.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",

                                        ] )
def test_hope_benchmarks( test_file ):
    pyccel_test(test_file)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import.py",
                                        "scripts/import_syntax/from_mod_import_as.py",
                                        "scripts/import_syntax/import_mod.py",
                                        "scripts/import_syntax/import_mod_as.py",
                                        "scripts/import_syntax/from_mod_import_func.py",
                                        "scripts/import_syntax/from_mod_import_as_func.py",
                                        "scripts/import_syntax/import_mod_func.py",
                                        "scripts/import_syntax/import_mod_as_func.py",
                                        "scripts/import_syntax/collisions.py"
                                        ] )
def test_import_syntax( test_file ):
    pyccel_test(test_file)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_user.py",
                                        "scripts/import_syntax/from_mod_import_as_user.py",
                                        "scripts/import_syntax/import_mod_user.py",
                                        "scripts/import_syntax/import_mod_as_user.py",
                                        "scripts/import_syntax/from_mod_import_user_func.py",
                                        "scripts/import_syntax/from_mod_import_as_user_func.py",
                                        "scripts/import_syntax/import_mod_user_func.py",
                                        "scripts/import_syntax/import_mod_as_user_func.py",
                                        ] )
def test_import_syntax_user( test_file ):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py")

#------------------------------------------------------------------------------
def test_numpy_kernels_compile():
    cwd = get_abs_path(".")
    compile_pyccel(os.path.join(cwd, "scripts/numpy/"), "numpy_kernels.py")

#------------------------------------------------------------------------------
def test_multiple_results():
    pyccel_test("scripts/runtest_multiple_results.py",
            dependencies = "scripts/default_args_mod.py",
            output_dtype = [int,float,complex,bool,int,complex,
                int,bool,float,float,float,float,float,float,
                float,float,float,float,float,float,
                float,float,float,float,float,float])
