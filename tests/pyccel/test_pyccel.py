# pylint: disable=missing-function-docstring, missing-module-docstring
import subprocess
import json
import os
import shutil
import sys
import re
import random
import pytest
import numpy as np
from pyccel.codegen.pipeline import execute_pyccel
from pyccel.ast.utilities import python_builtin_libs

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
        assert(os.path.isfile(exefile1))
        return exefile1

#------------------------------------------------------------------------------
def insert_pyccel_folder(abs_path):
    base_dir = os.path.dirname(abs_path)
    base_name = os.path.basename(abs_path)
    return os.path.join(base_dir, "__pyccel__" + os.environ.get('PYTEST_XDIST_WORKER', ''), base_name)

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
def compile_pyccel(path_dir, test_file, options = ""):
    if "python" in options and "--output" not in options:
        options += " --output=__pyccel__"
    cmd = [shutil.which("pyccel"), "%s" % test_file]
    if options != "":
        cmd += options.strip().split()
    p = subprocess.Popen(cmd, universal_newlines=True, cwd=path_dir)
    p.wait()
    assert(p.returncode==0)

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
    gcc = shutil.which('gcc')
    folder = os.path.join(os.path.dirname(test_file), '__pyccel__')
    deps = []
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    for f in subfolders:
        for fi in os.listdir(f):
            root, ext = os.path.splitext(fi)
            if ext == '.c':
                deps.append(os.path.join(f, root) +'.py')
                with subprocess.Popen([gcc, '-c', fi, '-o', root+'.o'], text=True, cwd=f) as p:
                    p.wait()
    compile_fortran_or_c(gcc, '.c', path_dir, test_file, dependencies, deps, is_mod)

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
    compile_fortran_or_c(shutil.which('gfortran'), '.f90', path_dir, test_file, dependencies, (), is_mod)

#------------------------------------------------------------------------------
def compile_fortran_or_c(compiler, extension, path_dir, test_file, dependencies, std_deps, is_mod=False):
    """
    Compile Fortran or C code manually.

    Compile Fortran or C code manually. This is necessary when support is missing for the
    wrapper or when dependencies also need to be translated and compiled.

    Parameters
    ----------
    compiler : str
        The compiler (gfortran/gcc).

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
    root = insert_pyccel_folder(test_file)[:-3]

    assert(os.path.isfile(root+extension))

    deps = [dependencies] if isinstance(dependencies, str) else dependencies
    base_dir = os.path.dirname(root)
    if not is_mod:
        base_name = os.path.basename(root)
        prog_root = os.path.join(base_dir, "prog_"+base_name)
        if os.path.isfile(prog_root+extension):
            compile_fortran_or_c(compiler, extension,
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
        command.append("%s.o" % root)
    else:
        command.append("%s" % test_file[:-3])

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
        assert(p.returncode==0)
        return out

#------------------------------------------------------------------------------
def get_value(string, regex, conversion):
    match = regex.search(string)
    assert(match)
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
        assert(p_list==f_list)
    elif dtype is complex:
        rx = re.compile('[-0-9.eEj]+')
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
            rx = re.compile('[-0-9.eE]+')
            f, f_output  = get_value(f_output, rx, float)
            f2, f_output = get_value(f_output, rx, float)
            f = f+f2*1j
        assert(np.isclose(p, f))
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
        assert(np.isclose(p, f))

    elif dtype is int:
        rx = re.compile('[-0-9eE]+')
        p, p_output = get_value(p_output, rx, int)
        f, f_output = get_value(f_output, rx, int)
        assert(p==f)
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
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

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
    assert(p.returncode==0)

    compare_pyth_fort_output(pyth_out, fort_out)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_funcs(language):
    pyccel_test("scripts/runtest_funcs.py", language = language)

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
# See issue #756 for c problem
def test_generic_functions():
    pyccel_test("scripts/runtest_generic_functions.py",
            dependencies = "scripts/generic_functions.py",
            compile_with_pyccel = False,
            output_dtype = [float, float, float, float, float, float,
                    float, float, float, float, float, float, float, int, float,
                    int, int])

#------------------------------------------------------------------------------
def test_default_arguments(language):
    pyccel_test("scripts/runtest_default_args.py",
            dependencies = "scripts/default_args_mod.py",
            output_dtype = [int, int, float, float, float,
                float, float, float, float, bool, bool, bool,
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
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/hope_fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/hope_pisum.py",
                                        "scripts/hope_benchmarks/hope_ln_python.py",
                                        "scripts/hope_benchmarks/hope_pairwise_python.py",
                                        "scripts/hope_benchmarks/point_spread_func.py",
                                        "scripts/hope_benchmarks/simplify.py",
                                        "scripts/hope_benchmarks_decorators/fib.py",
                                        "scripts/hope_benchmarks_decorators/hope_ln_python.py",
                                        "scripts/hope_benchmarks_decorators/hope_pairwise_python.py",
                                        "scripts/hope_benchmarks_decorators/point_spread_func.py",
                                        "scripts/hope_benchmarks_decorators/simplify.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",

                                        ] )
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
def test_hope_benchmarks( test_file, language ):
    pyccel_test(test_file, language=language)

#------------------------------------------------------------------------------
@pytest.mark.c
@pytest.mark.parametrize( "test_file", ["scripts/hope_benchmarks/hope_fib.py",
                                        "scripts/hope_benchmarks/quicksort.py",
                                        "scripts/hope_benchmarks/hope_pisum.py",
                                        "scripts/hope_benchmarks/hope_ln_python.py",
                                        "scripts/hope_benchmarks/hope_pairwise_python.py",
                                        pytest.param("scripts/hope_benchmarks/point_spread_func.py",
                                            marks = pytest.mark.skip(reason="Numpy sum not implemented in c")),
                                        "scripts/hope_benchmarks/simplify.py",
                                        "scripts/hope_benchmarks_decorators/fib.py",
                                        "scripts/hope_benchmarks_decorators/hope_ln_python.py",
                                        "scripts/hope_benchmarks_decorators/hope_pairwise_python.py",
                                        pytest.param("scripts/hope_benchmarks_decorators/point_spread_func.py",
                                            marks = pytest.mark.skip(reason="Numpy sum not implemented in c")),
                                        "scripts/hope_benchmarks_decorators/simplify.py",
                                        "scripts/hope_benchmarks_decorators/quicksort.py",
                                        ] )
def test_hope_benchmarks_c( test_file ):
    pyccel_test(test_file, language='c')

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
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
def test_import_syntax( test_file, language ):
    pyccel_test(test_file, language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/import_syntax/from_mod_import_as_user_func.py",
                                        "scripts/import_syntax/from_mod_import_as_user.py",
                                        "scripts/import_syntax/collisions2.py"
                                        ] )
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
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
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
@pytest.mark.xdist_incompatible
def test_import_syntax_user( test_file, language ):
    pyccel_test(test_file, dependencies = "scripts/import_syntax/user_mod.py", language = language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "language", (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
    )
)
@pytest.mark.xdist_incompatible
def test_import_collisions(language):
    pyccel_test("scripts/import_syntax/collisions4.py",
            dependencies = ["scripts/import_syntax/user_mod.py", "scripts/import_syntax/user_mod2.py"],
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
                float, float, float, float, float, float
                , float, float, float, float, int, int], language=language)

#------------------------------------------------------------------------------
def test_elemental(language):
    pyccel_test("scripts/decorators_elemental.py", language = language)

#------------------------------------------------------------------------------
def test_print_strings(language):
    types = str
    pyccel_test("scripts/print_strings.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_print_integers(language):
    types = str
    pyccel_test("scripts/print_integers.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_print_tuples(language):
    types = str
    pyccel_test("scripts/print_tuples.py", language=language, output_dtype=types)

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
def test_arrays_view(language):
    types = [int] * 10 + [int] * 10 + [int] * 4 + [int] * 4 + [int] * 10 + \
            [int] * 6 + [int] * 10 + [int] * 10 + [int] * 25 + [int] * 60
    pyccel_test("scripts/arrays_view.py", language=language, output_dtype=types)

#------------------------------------------------------------------------------
def test_return_numpy_arrays(language):
    types = [int]*4 # 4 ints for a
    types += [int]*2 # 2 ints for b
    types += [float]*2 # 2 floats for c
    types += [bool]*2 # 2 bools for d
    types += [complex]*2 # 2 complexs for e
    types += [float]*5 # 5 floats for h
    types += [int]*5 # 5 ints for g
    types += [int]*4 # 4 ints for k
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
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = pytest.mark.fortran)
        # Test does not make sense in pure python
    )
)
def test_headers(language):
    test_file = "scripts/runtest_headers.py"
    test_file = os.path.normpath(test_file)
    test_file = get_abs_path(test_file)

    header_file = 'scripts/headers.pyh'
    header_file = os.path.normpath(header_file)
    header_file = get_abs_path(header_file)

    with open(test_file, 'w') as f:
        code = ("from headers import f\n"
                "def f(x):\n"
                "    y = x\n"
                "    return y\n"
                "if __name__ == '__main__':\n"
                "    print(f(1))\n")

        f.write(code)

    with open(header_file, 'w') as f:
        code =("#$ header metavar ignore_at_import=True\n"
               "#$ header function f(int)")

        f.write(code)

    test_file = os.path.normpath(test_file)
    cwd = os.path.dirname(test_file)
    cwd = get_abs_path(cwd)

    pyccel_commands = " --language="+language

    compile_pyccel(cwd, test_file, pyccel_commands)

    lang_out = get_lang_output(test_file, language)
    assert int(lang_out) == 1

    with open(test_file, 'w') as f:
        code = ("from headers import f\n"
                "def f(x):\n"
                "    y = x\n"
                "    return y\n"
                "if __name__ == '__main__':\n"
                "    print(f(1.5))\n")

        f.write(code)

    with open(header_file, 'w') as f:
        code =("#$ header metavar ignore_at_import=True\n"
               "#$ header function f(float)")

        f.write(code)

    compile_pyccel(cwd, test_file, pyccel_commands)

    lang_out = get_lang_output(test_file, language)
    assert float(lang_out) == 1.5

    with open(test_file, 'w') as f:
        code = ("")
        f.write(code)

#------------------------------------------------------------------------------
def test_basic_header():
    filename='scripts/basic_header.pyh'
    cwd = get_abs_path('.')
    compile_pyccel(cwd, filename)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/classes/classes.py",
                                        "scripts/classes/classes_1.py",
                                        "scripts/classes/classes_5.py",
                                        "scripts/classes/generic_methods.py",
                                        ] )
@pytest.mark.parametrize( 'language', (
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("fortran", marks = pytest.mark.fortran)
    )
)

def test_classes_f_only( test_file , language):
    if language == "python":
        pyccel_test(test_file, language=language)
    else:
        pyccel_test(test_file, compile_with_pyccel = False, language=language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.parametrize( "test_file", ["scripts/classes/classes_2_C.py",
                                        ] )
@pytest.mark.parametrize( 'language', (
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = pytest.mark.fortran)
    )
)

def test_classes( test_file , language):
    if language == "python":
        pyccel_test(test_file, language=language)
    else:
        pyccel_test(test_file, compile_with_pyccel = False, language=language)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/lapack_subroutine.py",
                                        ] )
@pytest.mark.skipif( sys.platform == 'win32', reason="Compilation problem. On execution Windows raises: error while loading shared libraries: liblapack.dll: cannot open shared object file: No such file or directory" )
@pytest.mark.external
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

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Collisions (initialised boolean) are not handled."),
            pytest.mark.c]
        )
    )
)
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
                                        "scripts/asserts/unvalid_assert1.py",
                                        "scripts/asserts/unvalid_assert2.py",
                                        "scripts/asserts/unvalid_assert3.py",
                                        ] )

def test_assert(language, test_file):
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
    assert (not lang_out and not pyth_out) or (lang_out and pyth_out)

#------------------------------------------------------------------------------
@pytest.mark.parametrize( "test_file", ["scripts/exits/empty_exit.py",
                                        "scripts/exits/negative_exit1.py",
                                        "scripts/exits/negative_exit2.py",
                                        "scripts/exits/positive_exit1.py",
                                        "scripts/exits/positive_exit2.py",
                                        "scripts/exits/positive_exit3.py",
                                        "scripts/exits/zero_exit.py",
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
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Collisions are not handled. And chained imports (see #756)"),
            pytest.mark.c]
        )
    )
)
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

def test_function_aliasing():
    pyccel_test("scripts/runtest_function_alias.py",
            language = 'fortran')

#------------------------------------------------------------------------------

def test_function(language):
    pyccel_test("scripts/functions.py",
            language = language, output_dtype=[str]+[int]*7 )

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
def test_inline(language):
    pyccel_test("scripts/decorators_inline.py", language = language)

#------------------------------------------------------------------------------
@pytest.mark.xdist_incompatible
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Collisions (initialised boolean) are not handled."),
            pytest.mark.c]
        )
    )
)
def test_inline_import(language):
    pyccel_test("scripts/runtest_decorators_inline.py",
            dependencies = ("scripts/decorators_inline.py"),
                language = language)

#------------------------------------------------------------------------------
def test_json():
    pyccel_test("scripts/runtest_funcs.py", language = 'fortran',
            pyccel_commands='--export-compile-info test.json')
    with open(get_abs_path('scripts/test.json'), 'r') as f:
        dict_1 = json.load(f)
    pyccel_test("scripts/runtest_funcs.py", language = 'fortran',
        pyccel_commands='--compiler test.json --export-compile-info test2.json')
    with open(get_abs_path('scripts/test2.json'), 'r') as f:
        dict_2 = json.load(f)

    assert dict_1 == dict_2

#------------------------------------------------------------------------------
def test_reserved_file_name():
    with pytest.raises(ValueError) as exc_info:
        libname = str(random.choice(tuple(python_builtin_libs))) + ".py" # nosec B311
        execute_pyccel(fname=libname)
    assert str(exc_info.value) == f"File called {libname} has the same name as a Python built-in package and can't be imported from Python. See #1402"

def test_concatentation():
    pyccel_test("scripts/concatenation.py",
                language = 'fortran',
                output_dtype=[int]*15+[str])
