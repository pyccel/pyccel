# pylint: disable=missing-function-docstring, missing-module-docstring
import subprocess
import json
import os
import platform
import shutil
import sys
import re
import random
import pytest
import numpy as np
from filelock import FileLock
from pyccel.codegen.pipeline import execute_pyccel
from pyccel.ast.utilities import python_builtin_libs
from pyccel.compilers.default_compilers import available_compilers

#==============================================================================
# UTILITIES
#==============================================================================

#------------------------------------------------------------------------------

def get_abs_path(relative_path):
    relative_path = os.path.normpath(relative_path)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_dir, relative_path)

#------------------------------------------------------------------------------
def get_exe(filename, language=None):
    if language!="python":
        exefile1 = os.path.splitext(filename)[0]
    else:
        exefile1 = filename

    if sys.platform == "win32" and language!="python":
        exefile1 += ".exe"

    dirname = os.path.dirname(filename)
    basename = os.path.basename(exefile1)
    exefile2 = os.path.join(dirname, basename)

    if os.path.isfile(exefile2):
        return exefile2
    else:
        assert os.path.isfile(exefile1)
        return exefile1

#------------------------------------------------------------------------------
def insert_pyccel_folder(abs_path):
    base_dir = os.path.dirname(abs_path)
    base_name = os.path.basename(abs_path)
    return os.path.join(base_dir, "__pyccel__" + os.environ.get('PYTEST_XDIST_WORKER', ''), base_name)

#------------------------------------------------------------------------------
def get_python_output(abs_path, cwd = None):
    with subprocess.Popen([sys.executable , abs_path], stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd) as p:
        out, _ = p.communicate()
        assert p.returncode==0
    return out

#------------------------------------------------------------------------------
def compile_pyccel(path_dir, test_file, options = ""):
    if "python" in options and "--output" not in options:
        options += " --output=__pyccel__"
    cmd = [shutil.which("pyccel"), test_file]
    if options != "":
        cmd += options.strip().split()
    p = subprocess.Popen(cmd, universal_newlines=True, cwd=path_dir)
    p.wait()
    assert p.returncode==0

#------------------------------------------------------------------------------
def compile_c(path_dir, test_file, dependencies, is_mod=False):
    """
    Compile C code manually.

    Compile C code manually. This is a wrapper around compile_fortran_or_c.

    Parameters
    ----------
    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated.

    dependencies : list of str
        A list of any Python dependencies of the file.

    is_mod : bool, default=False
        True if translating a module, False if translating a program

    See also
    --------
    compile_fortran_or_c : The function that is called.
    """
    compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    compiler_info = available_compilers[compiler_family]['c']
    compiler = compiler_info['exec']
    folder = os.path.join(os.path.dirname(test_file), '__pyccel__')
    deps = []
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    for f in subfolders:
        for fi in os.listdir(f):
            root, ext = os.path.splitext(fi)
            if ext == '.c':
                deps.append(os.path.join(f, root) +'.py')
                with subprocess.Popen([compiler, '-c', fi, '-o', root+'.o'], text=True, cwd=f) as p:
                    p.wait()
    compile_fortran_or_c(compiler_info, '.c', path_dir, test_file, dependencies, deps, is_mod)

#------------------------------------------------------------------------------
def compile_fortran(path_dir, test_file, dependencies, is_mod=False):
    """
    Compile Fortran code manually.

    Compile Fortran code manually. This is a wrapper around compile_fortran_or_c.

    Parameters
    ----------
    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated.

    dependencies : list of str
        A list of any Python dependencies of the file.

    is_mod : bool, default=False
        True if translating a module, False if translating a program

    See also
    --------
    compile_fortran_or_c : The function that is called.
    """
    compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    compiler_info = available_compilers[compiler_family]['fortran']
    compile_fortran_or_c(compiler_info, '.f90', path_dir, test_file, dependencies, (), is_mod)

#------------------------------------------------------------------------------
def compile_fortran_or_c(compiler_info, extension, path_dir, test_file, dependencies, std_deps, is_mod=False):
    """
    Compile Fortran or C code manually.

    Compile Fortran or C code manually. This is necessary when support is missing for the
    wrapper or when dependencies also need to be translated and compiled.

    Parameters
    ----------
    compiler_info : dict
        A dictionary describing the compiler properties.

    extension : str
        The extension of the generated file (.c/.f90).

    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated.

    dependencies : list of str
        A list of any Python dependencies of the file.

    std_deps : list of str
        A list of any language-specific dependencies of the file (e.g. ndarrays).

    is_mod : bool, default=False
        True if translating a module, False if translating a program
    """
    compiler = compiler_info['exec']
    root = insert_pyccel_folder(test_file)[:-3]

    assert os.path.isfile(root+extension)

    deps = [dependencies] if isinstance(dependencies, str) else dependencies
    base_dir = os.path.dirname(root)
    if not is_mod:
        base_name = os.path.basename(root)
        prog_root = os.path.join(base_dir, "prog_"+base_name)
        if os.path.isfile(prog_root+extension):
            compile_fortran_or_c(compiler_info, extension,
                                path_dir, test_file,
                                dependencies, std_deps,
                                is_mod = True)
            root = prog_root
            deps.append(test_file)

    if is_mod:
        command = [shutil.which(compiler), "-c", root+extension]
        for d in deps:
            d = insert_pyccel_folder(d)
            command.append("-I"+os.path.dirname(d))
        for d in std_deps:
            command.append("-I"+os.path.dirname(d))
    else:
        command = [compiler, "-O3", root+extension]
        for d in deps:
            d = insert_pyccel_folder(d)
            command.append(d[:-3]+".o")
            command.append("-I"+os.path.dirname(d))
        for d in std_deps:
            command.append(d[:-3]+".o")
            command.append("-I"+os.path.dirname(d))
    command.append("-I"+base_dir)

    command.append("-o")
    if is_mod:
        command.append(f"{root}.o")
    else:
        command.append(test_file[:-3])

    if 'module_output_flag' in compiler_info:
        command.append(compiler_info['module_output_flag'])
        command.append(base_dir)

    with subprocess.Popen(command, universal_newlines=True, cwd=path_dir) as p:
        p.wait()

#------------------------------------------------------------------------------
def get_lang_output(abs_path, language):
    abs_path = get_exe(abs_path, language)
    if language=="python":
        return get_python_output(abs_path)
    else:
        p = subprocess.Popen(["%s" % abs_path], stdout=subprocess.PIPE, universal_newlines=True)
        out, _ = p.communicate()
        assert p.returncode==0
        return out

#------------------------------------------------------------------------------
def get_value(string, regex, conversion):
    match = regex.search(string)
    assert match
    value = conversion(match.group())
    string = string[match.span()[1]:]
    return value, string

def compare_pyth_fort_output_by_type( p_output, f_output, dtype=float, language=None):

    if dtype is str:
        p_output_split = re.split('\n', p_output)
        f_output_split = re.split('\n', f_output)
        p_list = p_output_split[0].strip()
        f_list = f_output_split[0].strip()
        p_output = '\n'.join(p_output_split[1:])
        f_output = '\n'.join(f_output_split[1:])
        assert p_list==f_list
    elif dtype is complex:
        rx = re.compile('-?[0-9.]+([eE][+-]?[0-9]+)?j?')
        p, p_output = get_value(p_output, rx, complex)
        if p.imag == 0:
            p2, p_output = get_value(p_output, rx, complex)
            p = p+p2
        if language == 'python':
            f, f_output = get_value(f_output, rx, complex)
            if f.imag == 0:
                f2, f_output = get_value(f_output, rx, complex)
                f = f+f2
        else:
            rx = re.compile('-?[0-9.]+([eE][+-]?[0-9]+)?')
            f, f_output  = get_value(f_output, rx, float)
            f2, f_output = get_value(f_output, rx, float)
            f = f+f2*1j
        assert np.isclose(p, f)
    elif dtype is bool:
        rx = re.compile('TRUE|True|true|1|T|t|FALSE|False|false|F|f|0')
        bool_conversion = lambda m: m.lower() in ['true', 't', '1']
        p, p_output = get_value(p_output, rx, bool_conversion)
        f, f_output = get_value(f_output, rx, bool_conversion)
        assert p==f

    elif dtype is float:
        rx = re.compile('-?[0-9.]+([eE][+-]?[0-9]+)?')
        p, p_output = get_value(p_output, rx, float)
        f, f_output = get_value(f_output, rx, float)
        assert np.isclose(p, f)

    elif dtype is int:
        rx = re.compile('-?[0-9]+([eE][+-]?[0-9]+)?')
        p, p_output = get_value(p_output, rx, int)
        f, f_output = get_value(f_output, rx, int)
        assert p==f
    else:
        raise NotImplementedError("Type comparison not implemented")
    return p_output, f_output

#------------------------------------------------------------------------------
def compare_pyth_fort_output( p_output, f_output, dtype=float, language=None):

    if isinstance(dtype, list):
        for d in dtype:
            p_output, f_output = compare_pyth_fort_output_by_type(p_output, f_output, d, language=language)
    elif dtype is complex:
        while len(p_output)>0 and len(f_output)>0:
            p_output, f_output = compare_pyth_fort_output_by_type(p_output, f_output, complex, language=language)
    elif dtype is str:
        compare_pyth_fort_output_by_type(p_output, f_output, dtype)
    else:
        p_output = p_output.strip().split()
        f_output = f_output.strip().split()
        for p, f in zip(p_output, f_output):
            compare_pyth_fort_output_by_type(p, f, dtype)

#------------------------------------------------------------------------------
def pyccel_test(test_file, dependencies = None, compile_with_pyccel = True,
        cwd = None, pyccel_commands = "", output_dtype = float,
        language = None, output_dir = None):
    """
    Run pyccel and compare the output to ensure that the results
    are equivalent

    Parameters
    ----------
    test_file : str
                The name of the file containing the program.
                The path must either be absolute or relative
                to the folder containing this file
    dependencies : str/list
                The name of any files which are called by the
                test_file and must therefore be pyccelized in
                order to run it
                The paths must either be absolute or relative
                to the folder containing this file
    compile_with_pyccel : bool
                Indicates whether the compilation step should
                be handled by a basic call to gfortran/gcc (False)
                or internally by pyccel (True)
                default : True
    cwd : str
                The directory from which pyccel and other executables
                will be called
                default : The folder containing the test_file
    pyccel_commands : str
                Any additional commands which should be passed to
                pyccel
    output_dtype : type/list of types
                The types expected as output of the program.
                If one argument is provided then all types are
                assumed to be the same
    language : str
                The language pyccel should translate to
                default = 'fortran'
    output_dir : str
                The folder in which the generated files should be
                saved
    """

    rel_test_dir = os.path.dirname(test_file)

    test_file = os.path.normpath(test_file)

    if (cwd is None):
        cwd = os.path.dirname(test_file)

    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)

    pyth_out = get_python_output(test_file, cwd)

    if language:
        pyccel_commands += " --language="+language
    else:
        language='fortran'

    if output_dir is None:
        if language=="python":
            output_dir = os.path.join(get_abs_path(rel_test_dir), '__pyccel__')

    if dependencies:
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        for i, d in enumerate(dependencies):
            dependencies[i] = get_abs_path(d)
            if output_dir:
                rel_path = os.path.relpath(os.path.dirname(d), start=rel_test_dir)
                output = get_abs_path(os.path.join(output_dir, rel_path))
                pyc_command = pyccel_commands + ' --output={}'.format(output)
            else:
                pyc_command = pyccel_commands

            if not compile_with_pyccel:
                compile_pyccel (cwd, dependencies[i], pyc_command+" -t")
                if language == 'fortran':
                    compile_fortran(cwd, dependencies[i], [], is_mod = True)
                elif language == 'c':
                    compile_c(cwd, dependencies[i], [], is_mod = True)
            else:
                compile_pyccel(cwd, dependencies[i], pyc_command)

    if output_dir:
        pyccel_commands += " --output "+output_dir
        output_test_file = os.path.join(output_dir, os.path.basename(test_file))
    else:
        output_test_file = test_file

    if compile_with_pyccel:
        compile_pyccel(cwd, test_file, pyccel_commands)
    else:
        compile_pyccel (cwd, test_file, pyccel_commands+" -t")
        if not dependencies:
            dependencies = []
        if language=='fortran':
            compile_fortran(cwd, output_test_file, dependencies)
        elif language == 'c':
            compile_c(cwd, output_test_file, dependencies)

    lang_out = get_lang_output(output_test_file, language)
    compare_pyth_fort_output(pyth_out, lang_out, output_dtype, language)

#==============================================================================
# UNIT TESTS
#==============================================================================
@pytest.mark.xdist_incompatible
def test_relative_imports_in_project(language):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_rel_imports")
    dependencies = ['project_rel_imports/project/folder1/mod1.py',
                    'project_rel_imports/project/folder2/mod2.py',
                    'project_rel_imports/project/folder2/mod3.py']
    pyccel_test("project_rel_imports/runtest.py", dependencies,
            cwd = path_dir,
            language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_absolute_imports_in_project(language):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_abs_imports")
    dependencies = ['project_abs_imports/project/folder1/mod1.py',
             'project_abs_imports/project/folder2/mod2.py',
             'project_abs_imports/project/folder2/mod3.py']
    pyccel_test("project_abs_imports/runtest.py", dependencies,
            cwd = path_dir,
            language = language)

#------------------------------------------------------------------------------
def test_rel_imports_python_accessible_folder(language):
    # pyccel is called on scripts/folder2/runtest_rel_imports.py from the scripts folder
    # From this folder python understands relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    from scripts.folder2.runtest_rel_imports import test_func

    tmp_dir = os.path.join(base_dir, '__pyccel__')

    pyth_out = str(test_func())

    pyccel_opt = '--language={}'.format(language)
    if language == 'python':
        pyccel_opt += ' --output={}'.format(os.path.join(tmp_dir, "folder2"))
    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/folder2_funcs.py"), pyccel_opt)
    compile_pyccel(path_dir, get_abs_path("scripts/folder2/runtest_rel_imports.py"), pyccel_opt)
    if language == 'python':
        test_location = "__pyccel__.folder2.runtest_rel_imports"
    else:
        test_location = "scripts.folder2.runtest_rel_imports"
    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), test_location],
                stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert p.returncode==0

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_multi_imports_project(language):

    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "project_multi_imports")
    dependencies = ['project_multi_imports/file1.py',
             'project_multi_imports/file2.py',
             'project_multi_imports/file3.py']
    pyccel_test("project_multi_imports/file4.py", dependencies,
            cwd = path_dir,
            language = language,
            output_dtype = str)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports_compile(language):
    pyccel_test("scripts/runtest_imports.py", "scripts/funcs.py",
            compile_with_pyccel = False, language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports_in_folder(language):
    pyccel_test("scripts/runtest_folder_imports.py", "scripts/folder1/folder1_funcs.py",
            compile_with_pyccel = False, language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_imports(language):
    pyccel_test("scripts/runtest_imports.py", "scripts/funcs.py",
            language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_folder_imports(language):
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    tmp_dir = os.path.join(base_dir, '__pyccel__')

    from scripts.folder2.runtest_imports2 import test_func
    pyth_out = str(test_func())

    language_opt = '--language={}'.format(language)
    pyccel_opt = language_opt
    if language == 'python':
        pyccel_opt = language_opt+' --output={}'.format(os.path.join(tmp_dir, "folder1"))
    compile_pyccel(os.path.join(path_dir, "folder1"), get_abs_path("scripts/folder1/folder1_funcs.py"),
            pyccel_opt)
    if language == 'python':
        pyccel_opt = language_opt+' --output={}'.format(os.path.join(tmp_dir, "folder2"))
    compile_pyccel(os.path.join(path_dir, "folder2"), get_abs_path("scripts/folder2/runtest_imports2.py"),
            pyccel_opt)

    if language == 'python':
        test_location = "__pyccel__.folder2.runtest_imports2"
    else:
        test_location = "scripts.folder2.runtest_imports2"
    p = subprocess.Popen([sys.executable , "%s" % os.path.join(base_dir, "run_import_function.py"), test_location],
            stdout=subprocess.PIPE, universal_newlines=True)
    fort_out, _ = p.communicate()
    assert p.returncode==0

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_funcs(language):
    pyccel_test("scripts/runtest_funcs.py", language = language)

@pytest.mark.xdist_incompatible
def test_capitalised_language(language):
    test_file = get_abs_path("scripts/runtest_funcs.py")
    cwd = os.path.dirname(test_file)
    output_folder = "__pyccel__" + os.environ.get('PYTEST_XDIST_WORKER', '')
    compile_pyccel(cwd, test_file, f'--language={language.capitalize()} --output={output_folder}')

#------------------------------------------------------------------------------
# Enumerate not supported in c
def test_inout_func(language):
    pyccel_test("scripts/runtest_inoutfunc.py", language = language)

#------------------------------------------------------------------------------
def test_bool(language):
    pyccel_test("scripts/bool_comp.py", output_dtype = bool, language = language)

#------------------------------------------------------------------------------
def test_expressions(language):
    types = [float, complex, int, float, float, int] + [float]*3 + \
            [complex, int, complex, complex, int, int, float] + [complex]*3 + \
            [float]*3 + [int] + [float]*2 + [int] + [float]*3 + [int] + \
            [float]*3 + [int]*2 + [float]*2 + [int]*5 + [complex] + [bool]*9
    pyccel_test("scripts/expressions.py", language=language,
            output_dtype = types)

#------------------------------------------------------------------------------
@pytest.mark.fortran
def test_generic_functions():
    # Only testing Fortran for simple compilation outside of Pyccel
    pyccel_test("scripts/runtest_generic_functions.py",
            dependencies = "scripts/generic_functions.py",
            compile_with_pyccel = False,
            output_dtype = [float, float, float, float, float, float,
                    float, float, float, float, float, float, float, int, float,
                    int, int])

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_default_arguments(language):
    pyccel_test("scripts/runtest_default_args.py",
            dependencies = "scripts/default_args_mod.py",
            output_dtype = [int, int, float, float, float,
                float, float, float, float, bool, bool, bool,
                float, float, float, float, int, int,
                float, float, float, float],
            language=language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_pyccel_calling_directory(language):
    cwd = get_abs_path(".")

    test_file = get_abs_path("scripts/runtest_funcs.py")
    pyth_out = get_python_output(test_file)

    language_opt = '--language={}'.format(language)
    compile_pyccel(cwd, test_file, language_opt)

    if language == "python":
        test_file = get_abs_path(os.path.join('__pyccel__',
                                os.path.basename(test_file)))
    fort_out = get_lang_output(test_file, language)

    compare_pyth_fort_output( pyth_out, fort_out )

#------------------------------------------------------------------------------
def test_in_specified(language):
    pyccel_test("scripts/runtest_degree_in.py", language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/hope_pisum.py",
                                        "scripts/hope_benchmarks/hope_ln_python.py",
                                        "scripts/hope_benchmarks/hope_pairwise_python.py",
                                        "scripts/hope_benchmarks/point_spread_func.py",
                                        "scripts/hope_benchmarks/simplify.py",
                                        ] )
def test_hope_benchmarks( test_file, language ):
    pyccel_test(test_file, language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import.py",
                                        "scripts/import_syntax/from_mod_import_as.py",
                                        "scripts/import_syntax/import_mod.py",
                                        "scripts/import_syntax/import_mod_as.py",
                                        "scripts/import_syntax/from_mod_import_func.py",
                                        "scripts/import_syntax/from_mod_import_as_func.py",
                                        "scripts/import_syntax/import_mod_func.py",
                                        "scripts/import_syntax/import_mod_as_func.py",
                                        "scripts/import_syntax/collisions3.py",
                                        "scripts/import_syntax/collisions5.py",
                                        ] )
def test_import_syntax(test_file, language):
    pyccel_test(test_file, language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_as_user_func.py",
                                        "scripts/import_syntax/from_mod_import_as_user.py",
                                        "scripts/import_syntax/collisions2.py",
                                        "scripts/runtest_import_mod_project_as.py",
                                        ] )
@pytest.mark.xdist_incompatible
def test_import_syntax_user_as( test_file, language ):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py",
            language = language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_user.py",
                                        "scripts/import_syntax/import_mod_user.py",
                                        "scripts/import_syntax/import_mod_as_user.py",
                                        "scripts/import_syntax/from_mod_import_user_func.py",
                                        "scripts/import_syntax/import_mod_user_func.py",
                                        "scripts/import_syntax/import_mod_as_user_func.py",
                                        ] )
@pytest.mark.xdist_incompatible
def test_import_syntax_user(test_file, language):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py", language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_import_collisions(language):
    pyccel_test("scripts/import_syntax/collisions4.py",
            dependencies = ["scripts/import_syntax/user_mod.py", "scripts/import_syntax/user_mod2.py"],
            language=language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_import_collisions_builtins(language):
    pyccel_test("scripts/import_syntax/collisions6.py",
            dependencies = ["scripts/import_syntax/user_mod_builtin_conflict.py"],
            language=language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_class_import_as(language):
    pyccel_test("scripts/import_syntax/from_cls_mod_import_as_user.py",
                dependencies = ["scripts/import_syntax/user_cls_mod.py"],
                language=language)

#------------------------------------------------------------------------------
# Numpy sum required
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
def test_numpy_kernels_compile(language):
    pyccel_opt = '--language={}'.format(language)
    cwd = get_abs_path(".")
    compile_pyccel(os.path.join(cwd, "scripts/numpy/"),
            "numpy_kernels.py",
            pyccel_opt)

#------------------------------------------------------------------------------
def test_multiple_results(language):
    pyccel_test("scripts/runtest_multiple_results.py",
            output_dtype = [int, float, complex, bool, int, complex,
                int, bool, float, float, float, float, float, float,
                float, float, float, float, float, float,
                float, float, float, float, float, float,
                float, float, float, float, int, int], language=language)

#------------------------------------------------------------------------------
def test_elemental(language):
    pyccel_test("scripts/decorators_elemental.py", language = language)

#------------------------------------------------------------------------------
def test_print_strings(language):
    types = str
    pyccel_test("scripts/print_strings.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="Can't print NaN in Fortran"),
            pytest.mark.fortran])
    )
)
def test_print_nan(language):
    types = str
    pyccel_test("scripts/print_nan.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_print_integers(language):
    types = str
    pyccel_test("scripts/print_integers.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_print_sp_and_end(language):
    types = str
    pyccel_test("scripts/print_sp_and_end.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_c_arrays(language):
    types = [int]*15 + [float]*5 + [int]*25 + [float]* 20 * 5 + \
            [complex] * 3 * 10 + [complex] * 5 + [float] * 10 + [float] * 6 + \
            [float] * 2 * 3 + [complex] * 3 * 10 + [float] * 2 * 3 + [int] * 3
    pyccel_test("scripts/c_arrays.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Negative slices are not handled"),
            pytest.mark.c]
        )
    )
)
def test_arrays_view(language):
    types = [int] * 10 + [int] * 10 + [int] * 4 + [int] * 4 + [int] * 10 + \
            [int] * 6 + [int] * 10 + [int] * 10 + [int] * 25 + [int] * 60
    if platform.system() in ('Darwin', 'Windows') and language=='fortran':
        # MacOS compiler incorrectly reports
        # Fortran runtime error: Index '4378074096' of dimension 2 of array 'a' outside of expected range (0:2)
        # At line 208 of file /Users/runner/work/pyccel/pyccel/tests/pyccel/scripts/__pyccel__/arrays_view.f90
        # x(0:) => a(1_i64:, merge(3_i64 + v, v, v < 0_i64))
        pyccel_test("scripts/arrays_view.py", language=language, output_dtype=types,
                    pyccel_commands="--no-debug")
    else:
        pyccel_test("scripts/arrays_view.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_return_numpy_arrays(language):
    types = [int]*4 # 4 ints for a
    types += [int]*2 # 2 ints for b
    types += [float]*2 # 2 floats for c
    types += [bool]*2 # 2 bools for d
    types += [complex]*2 # 2 complexes for e
    types += [float]*5 # 5 floats for h
    types += [int]*5 # 5 ints for g
    types += [int]*4 # 4 ints for k
    types += [float]*48 # 48 floats for x
    pyccel_test("scripts/return_numpy_arrays.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_array_binary_op(language):
    types = [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 8
    pyccel_test("scripts/array_binary_operation.py", language = language, output_dtype=types)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/classes/classes.py",
                                        "scripts/classes/classes_1.py",
                                        "scripts/classes/classes_2.py",
                                        "scripts/classes/classes_3.py",
                                        "scripts/classes/classes_4.py",
                                        "scripts/classes/classes_5.py",
                                        "scripts/classes/classes_6.py",
                                        "scripts/classes/classes_7.py",
                                        "scripts/classes/classes_8.py",
                                        "scripts/classes/classes_9.py",
                                        "scripts/classes/pep526.py",
                                        "scripts/classes/class_variables.py",
                                        "scripts/classes/class_temporary_in_constructor.py",
                                        "scripts/classes/class_with_non_target_array_arg.py",
                                        "scripts/classes/class_pointer.py",
                                        "scripts/classes/class_pointer_2.py",
                                        ] )
def test_classes( test_file , language):
    pyccel_test(test_file, language=language)

def test_class_magic(language):
    pyccel_test("scripts/classes/class_magic.py", language=language,
            output_dtype = [int]*6 + [bool]*2 + [int])

def test_tuples_in_classes(language):
    test_file = "scripts/classes/tuples_in_classes.py"
    pyccel_test(test_file, language=language, output_dtype = [float, float, float, bool, bool])

def test_classes_type_print(language):
    test_file = "scripts/classes/empty_class.py"

    rel_test_dir = os.path.dirname(test_file)

    test_file = os.path.normpath(test_file)

    cwd = os.path.dirname(test_file)
    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)

    pyccel_commands = " --language="+language

    if language=="python":
        output_dir = os.path.join(get_abs_path(rel_test_dir), '__pyccel__')
        pyccel_commands += " --output "+output_dir
        output_test_file = os.path.join(output_dir, os.path.basename(test_file))
    else:
        output_test_file = test_file

    compile_pyccel(cwd, test_file, pyccel_commands)

    lang_out = get_lang_output(output_test_file, language)

    rx = re.compile(r'\bA\b')
    assert rx.search(lang_out)

def test_class_inline_array(language):
    pyccel_test("scripts/classes/class_inline.py",
                dependencies = ["scripts/classes/importable.py"],
                language = language,
                output_dtype = float)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.parametrize( "test_file", ["scripts/classes/generic_methods.py",
                                        ] )
@pytest.mark.parametrize( 'language', (
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Issue #1595"),
            pytest.mark.fortran])
    )
)

def test_interfaces_in_classes( test_file , language):
    pyccel_test(test_file, language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/lapack_subroutine.py",
                                        ] )
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: liblapack.dll: cannot open shared object file: No such file or directory" )
@pytest.mark.external
@pytest.mark.fortran
def test_lapack( test_file ):
    #TODO: Uncomment this when dgetri can be expressed with scipy
    #pyccel_test(test_file)

    #TODO: Remove the rest of the function when dgetri can be expressed with scipy
    test_file = os.path.normpath(test_file)
    test_file = get_abs_path(test_file)

    cwd = get_abs_path('.')

    compile_pyccel(cwd, test_file)

    lang_out = get_lang_output(test_file, 'fortran')
    rx = re.compile('[-0-9.eE]+')
    lang_out_vals = []
    while lang_out:
        try:
            f, lang_out = get_value(lang_out, rx, float)
            lang_out_vals.append(f)
        except AssertionError:
            lang_out = None
    output_mat = np.array(lang_out_vals).reshape(4, 4)
    expected_output = np.eye(4)

    assert np.allclose(output_mat, expected_output, rtol=1e-14, atol=1e-15)

#------------------------------------------------------------------------------
def test_type_print( language ):
    pyccel_test("scripts/runtest_type_print.py",
                language = language, output_dtype=str)

def test_container_type_print(language):
    test_file = "scripts/runtest_array_type_print.py"

    rel_test_dir = os.path.dirname(test_file)

    test_file = os.path.normpath(test_file)

    cwd = os.path.dirname(test_file)
    cwd = get_abs_path(cwd)

    test_file = get_abs_path(test_file)

    pyccel_commands = " --language="+language

    if language=="python":
        output_dir = os.path.join(get_abs_path(rel_test_dir), '__pyccel__')
        pyccel_commands += " --output "+output_dir
        output_test_file = os.path.join(output_dir, os.path.basename(test_file))
    else:
        output_test_file = test_file

    compile_pyccel(cwd, test_file, pyccel_commands)

    lang_out = get_lang_output(output_test_file, language)

    rx = re.compile(r'\bnumpy.ndarray\b')
    assert rx.search(lang_out)

    if language!="python":
        rx = re.compile(r'\bfloat64\b')
        assert rx.search(lang_out)
#------------------------------------------------------------------------------

def test_module_init( language ):
    test_mod  = get_abs_path("scripts/module_init.py")
    test_prog = get_abs_path("scripts/runtest_module_init.py")

    output_dir   = get_abs_path('scripts/__pyccel__')
    output_test_file = os.path.join(output_dir, os.path.basename(test_prog))

    cwd = get_abs_path("scripts")

    pyccel_commands = "--language="+language
    if language=="python":
        if output_dir is None:
            pyccel_commands += "--output="+output_dir

    pyth_out = get_python_output(test_prog)

    compile_pyccel(cwd, test_mod, pyccel_commands)

    if language != "python":
        pyth_mod_out = get_python_output(test_prog, cwd)
        compare_pyth_fort_output(pyth_out, pyth_mod_out, str, language)

    compile_pyccel(cwd, test_prog, pyccel_commands)

    if language == 'python' :
        lang_out = get_lang_output(output_test_file, language)
    else:
        lang_out = get_lang_output(test_prog, language)

    compare_pyth_fort_output(pyth_out, lang_out, str, language)

#------------------------------------------------------------------------------
def get_lang_exit_value(abs_path, language, cwd=None):
    abs_path = get_exe(abs_path, language)
    if language == "python":
        if cwd is None:
            p = subprocess.Popen([sys.executable , abs_path])
        else:
            p = subprocess.Popen([sys.executable , abs_path], cwd=cwd)
    else:
        p = subprocess.Popen([abs_path])
    p.communicate()
    return p.returncode

@pytest.mark.parametrize( "test_file", ["scripts/asserts/valid_assert.py",
                                        "scripts/asserts/invalid_assert1.py",
                                        "scripts/asserts/invalid_assert2.py",
                                        "scripts/asserts/invalid_assert3.py",
                                        ] )

def test_assert(language, test_file):
    test_dir = os.path.dirname(test_file)
    test_file = get_abs_path(os.path.normpath(test_file))

    output_dir   = os.path.join(get_abs_path(test_dir), '__pyccel__')
    output_test_file = os.path.join(output_dir, os.path.basename(test_file))

    cwd = get_abs_path(test_dir)

    pyccel_commands = " --language="+language
    pyccel_commands += " --output="+ output_dir

    compile_pyccel(cwd, test_file, pyccel_commands)
    lang_out = get_lang_exit_value(output_test_file, language)
    pyth_out = get_lang_exit_value(test_file, "python")
    assert (not lang_out and not pyth_out) or (lang_out and pyth_out)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/exits/empty_exit.py",
                                        "scripts/exits/negative_exit1.py",
                                        "scripts/exits/negative_exit2.py",
                                        "scripts/exits/positive_exit1.py",
                                        "scripts/exits/positive_exit2.py",
                                        "scripts/exits/positive_exit3.py",
                                        "scripts/exits/zero_exit.py",
                                        "scripts/exits/error_message_exit.py",
                                        ] )

def test_exit(language, test_file):
    test_dir = os.path.dirname(test_file)
    test_file = get_abs_path(os.path.normpath(test_file))

    output_dir   = os.path.join(get_abs_path(test_dir), '__pyccel__')
    output_test_file = os.path.join(output_dir, os.path.basename(test_file))

    cwd = get_abs_path(test_dir)

    if not language:
        language = "fortran"
    pyccel_commands = " --language="+language
    pyccel_commands += " --output="+ output_dir

    compile_pyccel(cwd, test_file, pyccel_commands)
    lang_out = get_lang_exit_value(output_test_file, language)
    pyth_out = get_lang_exit_value(test_file, "python")
    assert lang_out == pyth_out

#------------------------------------------------------------------------------
def test_module_init_collisions( language ):
    test_mod  = get_abs_path("scripts/module_init2.py")
    test_prog = get_abs_path("scripts/runtest_module_init2.py")

    output_dir   = get_abs_path('scripts/__pyccel__')
    output_test_file = os.path.join(output_dir, os.path.basename(test_prog))

    cwd = get_abs_path("scripts")

    pyccel_commands = "--language="+language
    if language=="python":
        if output_dir is None:
            pyccel_commands += "--output="+output_dir

    pyth_out = get_python_output(test_prog)

    compile_pyccel(cwd, test_mod, pyccel_commands)
    compile_pyccel(cwd, test_prog, pyccel_commands)

    if language == 'python' :
        lang_out = get_lang_output(output_test_file, language)
    else:
        lang_out = get_lang_output(test_prog, language)

    compare_pyth_fort_output(pyth_out, lang_out, [float, float, float, int, float, float, float, int], language)

@pytest.mark.fortran
def test_function_aliasing():
    pyccel_test("scripts/runtest_function_alias.py",
            language = 'fortran')

#------------------------------------------------------------------------------

def test_function(language):
    pyccel_test("scripts/functions.py",
            language = language, output_dtype=str )

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel', reason="1671", language='fortran')
def test_inline(language):
    pyccel_test("scripts/decorators_inline.py", language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel', reason="1671", language='fortran')
def test_inline_import(language):
    pyccel_test("scripts/runtest_decorators_inline.py",
            dependencies = ("scripts/decorators_inline.py"),
                language = language)

#------------------------------------------------------------------------------
@pytest.mark.language_agnostic
def test_json():
    output_dir = get_abs_path(insert_pyccel_folder('scripts/'))
    cmd = [shutil.which("pyccel"), 'config', 'export', f'{output_dir}/test.json', '--compiler-family', 'intel']
    subprocess.run(cmd, check=True)
    with open(get_abs_path(f'{output_dir}/test.json'), 'r', encoding='utf-8') as f:
        dict_1 = json.load(f)
    assert dict_1['c']['exec'] == 'icx'
    cmd = [shutil.which("pyccel"),
           'config',
           'export',
           f'{output_dir}/test2.json',
           '--compiler-config',
           f'{output_dir}/test.json']
    subprocess.run(cmd, check=True)
    with open(get_abs_path(f'{output_dir}/test2.json'), 'r', encoding='utf-8') as f:
        dict_2 = json.load(f)

    assert dict_1 == dict_2

#------------------------------------------------------------------------------
@pytest.mark.language_agnostic
def test_ambiguous_json():
    #TODO: Remove in v2.3 when --export-compiler-config is deprecated
    output_dir = get_abs_path(insert_pyccel_folder('scripts/'))
    cmd = [shutil.which("pyccel"), '--export-compiler-config', f'{output_dir}/test']
    subprocess.run(cmd, check=True)
    with open(get_abs_path(f'{output_dir}/test.json'), 'r', encoding='utf-8') as f:
        dict_1 = json.load(f)
    cmd = [shutil.which("pyccel"),
           'config',
           'export',
           f'{output_dir}/test2.json',
           '--compiler-config',
           f'{output_dir}/test.json']
    subprocess.run(cmd, check=True)
    with open(get_abs_path(f'{output_dir}/test2.json'), 'r', encoding='utf-8') as f:
        dict_2 = json.load(f)

    assert dict_1 == dict_2

@pytest.mark.xdist_incompatible
@pytest.mark.language_agnostic
def test_json_relative_path():
    output_dir = get_abs_path(insert_pyccel_folder('scripts/'))
    cmd = [shutil.which("pyccel"), 'config', 'export', f'{output_dir}/test.json']
    subprocess.run(cmd, check=True)
    shutil.move(get_abs_path(f'{output_dir}/test.json'), get_abs_path('scripts/hope_benchmarks/test.json'))
    compile_pyccel(get_abs_path('scripts/hope_benchmarks'), "../runtest_funcs.py", '--compiler-config test.json')

#------------------------------------------------------------------------------
@pytest.mark.language_agnostic
def test_reserved_file_name():
    with pytest.raises(ValueError) as exc_info:
        libname = str(random.choice(tuple(python_builtin_libs))) + ".py" # nosec B311
        execute_pyccel(fname=libname)
    assert str(exc_info.value) == f"File called {libname} has the same name as a Python built-in package and can't be imported from Python. See #1402"

#------------------------------------------------------------------------------
@pytest.mark.skip(reason="List concatenation not yet implemented")
def test_concatenation(language):
    pyccel_test("scripts/concatenation.py",
                language = language,
                output_dtype=[int]*15+[str])

#------------------------------------------------------------------------------
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c)
    )
)
@pytest.mark.xdist_incompatible
def test_class_imports(language):
    cwd = get_abs_path('project_class_imports')

    test_file = get_abs_path('project_class_imports/runtest.py')

    pyth_out = get_python_output(test_file, cwd)

    compile_file = get_abs_path('project_class_imports/project/basics/Point_mod.py')
    compile_pyccel(cwd, compile_file, f"--language={language} --verbose")

    out1 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out1, float, 'python')

    compile_file = get_abs_path('project_class_imports/project/basics/Line_mod.py')
    compile_pyccel(cwd, compile_file, f"--language={language} --verbose")

    out2 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out2, float, 'python')

    compile_file = get_abs_path('project_class_imports/project/shapes/Square_mod.py')
    compile_pyccel(cwd, compile_file, f"--language={language} --verbose")

    out3 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out3, float, 'python')

    compile_file = get_abs_path('project_class_imports/runtest.py')
    compile_pyccel(cwd, compile_file, f"--language={language} --verbose")

    lang_out = get_lang_output(test_file, language)
    compare_pyth_fort_output(pyth_out, lang_out, float, language)

#------------------------------------------------------------------------------
def test_time_execution_flag(language):
    test_file  = get_abs_path("scripts/runtest_funcs.py")

    cwd = get_abs_path("scripts")

    cmd = [shutil.which("pyccel"), test_file, f"--language={language}", "--time-execution"]
    with subprocess.Popen(cmd, universal_newlines=True, cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        result, _ = p.communicate()

    result_lines = result.split('\n')
    assert 'Timers' in result_lines[0]
    assert 'Total' in result_lines[-2]
    for l in result_lines[1:-1]:
        assert ' : ' in l

#------------------------------------------------------------------------------
def test_module_name_containing_conflict(language):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")
    compile_pyccel(path_dir, get_abs_path("scripts/endif.py"), options = f"--language={language}")

    test_file = get_abs_path("scripts/runtest_badly_named_module.py")
    out1 = get_python_output(test_file)
    out2 = get_python_output(test_file)

    assert out1 == out2

#------------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == 'win32' and not np.__version__.startswith('2.'), reason="Integer mismatch with numpy 1.*")
def test_stubs(language):
    """
    This tests that a stub file is generated and ensures the stub files are
    still generated with the expected format. However it is not a good test.
    It prevents any changes being made to the output format and doesn't
    check that it can be parsed. This test should be replaced once stub files
    can be read.
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, "scripts")

    with open(get_abs_path(f"scripts/runtest_stub.{language}.pyi"), 'r', encoding="utf-8") as f:
        expected_pyi = f.read()

    wk_dir = get_abs_path("scripts/stub_test")
    with FileLock(wk_dir+'.lock'):
        compile_pyccel(path_dir, get_abs_path("scripts/runtest_stub.py"), options = f"--language={language} --output=stub_test")
        with open(get_abs_path(f"scripts/stub_test/__pyccel__{os.environ.get('PYTEST_XDIST_WORKER', '')}/runtest_stub.pyi"), 'r', encoding="utf-8") as f:
            generated_pyi = f.read()
        shutil.rmtree(wk_dir)

    assert expected_pyi == generated_pyi

#------------------------------------------------------------------------------
def test_builtin_container_print(language):
    pyccel_test("scripts/print_builtin_containers.py", output_dtype = str,
            language = language)

#------------------------------------------------------------------------------
def test_pyccel_generated_compilation_dependency(language):
    pyccel_test("scripts/runtest_pyccel_generated_compilation_dependency.py",
            dependencies = ["scripts/pyccel_generated_compilation_dependency.py"],
            output_dtype = int,
            language = language)

#------------------------------------------------------------------------------
def test_generated_name_collision(language):
    pyccel_test("scripts/GENERATED_NAME_COLLISION.py", output_dtype = int,
            language = language)

#------------------------------------------------------------------------------
def test_array_tuple_shape(language):
    pyccel_test("scripts/array_tuple_shape.py", output_dtype = int,
            language = language)

#------------------------------------------------------------------------------
def test_varargs(language):
    pyccel_test("scripts/runtest_varargs.py",
                language = language)

#------------------------------------------------------------------------------
@pytest.mark.python
def test_varkwargs():
    pyccel_test("scripts/runtest_varkwargs.py",
                language = 'python',
                output_dtype = str)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel', reason="1671", language='fortran')
def test_inline_using_import(language):
    test_file = "scripts/inlining/runtest_inline_using_import.py"
    pyccel_test(test_file,
                dependencies = ["scripts/inlining/my_func.py",
                                "scripts/inlining/my_other_func.py",
                                "scripts/inlining/inline_using_import.py"],
                language = language,
                output_dtype = float)

    if language != 'python':
        test_abspath = get_abs_path(test_file)

        cwd = os.path.dirname(test_abspath)
        pyth_out = get_python_output(test_abspath, cwd)
        lang_out = get_lang_output(os.path.splitext(test_abspath)[0], language)
        compare_pyth_fort_output(pyth_out, lang_out, float, language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel', reason="1671", language='fortran')
def test_inline_using_import_2(language):
    pyccel_test("scripts/inlining/runtest_inline_using_import_2.py",
                dependencies = ["scripts/inlining/my_func.py",
                                "scripts/inlining/my_other_func.py",
                                "scripts/inlining/inline_using_import.py"],
                language = language,
                output_dtype = float)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', None) == 'intel', reason="1671", language='fortran')
def test_inline_using_named_import(language):
    pyccel_test("scripts/inlining/runtest_inline_using_named_import.py",
                dependencies = ["scripts/inlining/my_func.py",
                                "scripts/inlining/my_func2.py",
                                "scripts/inlining/inline_using_named_import.py"],
                language = language,
                output_dtype = float)

#------------------------------------------------------------------------------
def test_classes_array_property(language):
    pyccel_test("scripts/classes/runtest_classes_array_property.py",
                dependencies = ["scripts/classes/classes_array_property.py"],
                language = language,
                output_dtype = float)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_classes_pointer_import(language):
    cwd = get_abs_path("scripts/classes")
    test_file = get_abs_path("scripts/classes/runtest_class_pointer_2.py")

    pyth_out = get_python_output(test_file, cwd)

    dependency = get_abs_path("scripts/classes/class_pointer_2.py")
    compile_pyccel(cwd, dependency, f"--language={language}")

    pyth_interface_out = get_python_output(test_file, cwd)
    assert pyth_out == pyth_interface_out

    compile_pyccel(cwd, test_file, f"--language={language}")

    lang_out = get_lang_output(test_file, language)
    compare_pyth_fort_output(pyth_out, lang_out, float, language)
