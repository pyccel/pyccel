# pylint: disable=missing-function-docstring, missing-module-docstring
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pluggy
import pytest

from pyccel.ast.utilities import python_builtin_libs
from pyccel.codegen.pipeline import execute_pyccel
from pyccel.compilers.default_compilers import available_compilers

# ==============================================================================
# UTILITIES
# ==============================================================================

expected_extensions = {"fortran": ".f90", "c": ".c", "python": ".py"}


# ------------------------------------------------------------------------------
def copy_to_isolated_dir(isolated_dir, rel_paths):
    """
    Copy files into an isolated directory, preserving their relative layout.

    Copy each of `rel_paths` (relative to the folder containing this file)
    into `isolated_dir`, preserving that same relative path, so files
    belonging to the same test land next to each other and relative/package
    imports between them keep working.

    Parameters
    ----------
    isolated_dir : Path
        The directory (exclusive to the current test) into which files
        should be copied.

    rel_paths : list of str/Path
        The paths of the files to copy, relative to the folder containing
        this file.

    Returns
    -------
    list of Path
        The absolute paths of the copies, inside `isolated_dir`, in the same
        order as `rel_paths`.
    """
    base_path = Path(__file__).resolve().parent
    new_abs_paths = []
    for rel_path in rel_paths:
        src = base_path / rel_path
        dst = isolated_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        new_abs_paths.append(dst)

    return new_abs_paths


# ------------------------------------------------------------------------------
def compile_pyccel(path_dir, test_file, options=""):
    """
    Run `pyccel compile` on a file.

    Run `pyccel compile` on a file with verbose output requested, so the
    paths of the files pyccel generated, and the executable it produced
    (for a full compile), can be recovered from stdout instead of having to
    be guessed by the caller. See `parse_generated_file` and
    `parse_generated_executable`.

    Parameters
    ----------
    path_dir : str
        The directory pyccel should be run from.

    test_file : str
        The file which should be compiled.

    options : str
        Any additional command-line options to pass to pyccel.

    Returns
    -------
    str
        The captured stdout of the `pyccel compile` command.
    """
    if "python" in options and "--output" not in options:
        options += " --output=__pyccel__"
    options += " -v"
    cmd = [shutil.which("pyccel"), "compile", test_file]
    if options != "":
        cmd += options.strip().split()
    p = subprocess.run(cmd, cwd=path_dir, text=True, capture_output=True, check=True)
    return p.stdout


# ------------------------------------------------------------------------------
def parse_generated_file(output, extension):
    """
    Get the path of the generated module file from a verbose pyccel output.

    Parse the stdout of a translate-only, verbose (`-v`) `pyccel compile` run
    (see `compile_pyccel`) and return the path of the generated module source
    file with the given extension, i.e. the file pyccel printed that does not
    have a `prog_` prefix. Header/stub files and, if present, the separate
    program-driver file are ignored.

    Parameters
    ----------
    output : str
        The captured stdout of a translate-only, verbose `pyccel compile` run.

    extension : str
        The extension of the generated file to find (e.g. `.f90`/`.c`).

    Returns
    -------
    Path
        The path of the generated module file.
    """
    for line in output.splitlines():
        if line.startswith(">>> Printing ::"):
            path = Path(line.split("::", 1)[1].strip())
            if path.suffix == extension and not path.name.startswith("prog_"):
                return path
    raise AssertionError(f"No generated {extension} file found in pyccel output")


# ------------------------------------------------------------------------------
def parse_generated_executable(output, language="fortran"):
    """
    Get the path of the language output from a verbose pyccel output.

    Parse the stdout of a verbose (`-v`), full (non-translate-only) `pyccel
    compile` run (see `compile_pyccel`) and return the path of the file that
    should be run to obtain the program's output: the executable pyccel
    produced, or, when translating to Python, the translated `.py` file
    pyccel copied to its final (`--output`) location, since no executable is
    produced in that case.

    Parameters
    ----------
    output : str
        The captured stdout of a verbose, full `pyccel compile` run.

    language : str
        The language pyccel translated to.

    Returns
    -------
    Path
        The path of the generated executable, or of the translated Python
        file.
    """
    if language == "python":
        for line in output.splitlines():
            if line.startswith("cp "):
                return Path(line.split()[-1])
        raise AssertionError("No copied Python file found in pyccel output")
    for line in output.splitlines():
        if line.startswith(">> Compiling executable ::"):
            return Path(line.split("::", 1)[1].strip())
    raise AssertionError("No generated executable found in pyccel output")


# ------------------------------------------------------------------------------
def compile_c(
    path_dir, test_file, generated_file, generated_dependencies, is_mod=False
):
    """
    Compile C code manually.

    Compile C code manually. This is a wrapper around compile_fortran_or_c.

    Parameters
    ----------
    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated. Used only to determine the
        name/location of the output executable.

    generated_file : Path
        The path, in its final folder, of the C file generated by pyccel
        from `test_file`.

    generated_dependencies : list of Path
        A list of the C files, in their final folders, generated by pyccel
        from the Python dependencies of the file.

    is_mod : bool, default=False
        True if translating a module, False if translating a program

    Returns
    -------
    Path or None
        The path of the generated executable, or None if a module (rather
        than a program) was compiled.

    See also
    --------
    compile_fortran_or_c : The function that is called.
    """
    compiler_family = os.environ.get("PYCCEL_DEFAULT_COMPILER", "GNU")
    compiler_info = available_compilers[compiler_family]["c"]
    compiler = compiler_info["exec"]
    folder = generated_file.parent
    deps = []
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    for f in subfolders:
        for fi in f.iterdir():
            if fi.suffix == ".c":
                deps.append(f / (fi.with_suffix(".py")))
                subprocess.run(
                    [compiler, "-c", fi.name, "-o", fi.stem + ".o"], cwd=f, check=True
                )
    return compile_fortran_or_c(
        compiler_info,
        path_dir,
        test_file,
        generated_file,
        generated_dependencies,
        deps,
        is_mod,
    )


# ------------------------------------------------------------------------------
def compile_fortran(
    path_dir, test_file, generated_file, generated_dependencies, is_mod=False
):
    """
    Compile Fortran code manually.

    Compile Fortran code manually. This is a wrapper around compile_fortran_or_c.

    Parameters
    ----------
    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated. Used only to determine the
        name/location of the output executable.

    generated_file : Path
        The path, in its final folder, of the Fortran file generated by
        pyccel from `test_file`.

    generated_dependencies : list of Path
        A list of the Fortran files, in their final folders, generated by
        pyccel from the Python dependencies of the file.

    is_mod : bool, default=False
        True if translating a module, False if translating a program

    Returns
    -------
    Path or None
        The path of the generated executable, or None if a module (rather
        than a program) was compiled.

    See also
    --------
    compile_fortran_or_c : The function that is called.
    """
    compiler_family = os.environ.get("PYCCEL_DEFAULT_COMPILER", "GNU")
    compiler_info = available_compilers[compiler_family]["fortran"]
    return compile_fortran_or_c(
        compiler_info,
        path_dir,
        test_file,
        generated_file,
        generated_dependencies,
        (),
        is_mod,
    )


# ------------------------------------------------------------------------------
def compile_fortran_or_c(
    compiler_info,
    path_dir,
    test_file,
    generated_file,
    dependencies,
    std_deps,
    is_mod=False,
):
    """
    Compile Fortran or C code manually.

    Compile Fortran or C code manually. This is necessary when support is missing for the
    wrapper or when dependencies also need to be translated and compiled.

    Parameters
    ----------
    compiler_info : dict
        A dictionary describing the compiler properties.

    path_dir : str
        The path to the directory where the compilation command should be run from.

    test_file : str
        The Python file which was translated. Used only to determine the
        name/location of the output executable.

    generated_file : Path
        The path, in its final folder, of the Fortran/C file (already carrying
        its .c/.f90 extension) generated by pyccel from `test_file`.

    dependencies : list of Path
        A list of the Fortran/C files (already carrying their .c/.f90
        extension), in their final folders, generated by pyccel from the
        Python dependencies of the file.

    std_deps : list of str
        A list of any language-specific dependencies of the file (e.g. ndarrays).

    is_mod : bool, default=False
        True if translating a module, False if translating a program

    Returns
    -------
    Path or None
        The path of the generated executable, or None if a module (rather
        than a program) was compiled.
    """
    compiler = compiler_info["exec"]
    extension = generated_file.suffix
    root = generated_file.with_suffix("")

    assert generated_file.is_file()

    deps = list(dependencies)
    base_dir = root.parent
    if not is_mod:
        prog_root = base_dir / ("prog_" + root.name)
        if prog_root.with_suffix(extension).is_file():
            compile_fortran_or_c(
                compiler_info,
                path_dir,
                test_file,
                generated_file,
                dependencies,
                std_deps,
                is_mod=True,
            )
            root = prog_root
            deps.append(generated_file)

    if is_mod:
        command = [shutil.which(compiler), "-c", str(root.with_suffix(extension))]
        for d in deps:
            command.append("-I" + str(d.parent))
        for d in std_deps:
            command.append("-I" + str(Path(d).parent))
    else:
        command = [compiler, "-O3", str(root.with_suffix(extension))]
        for d in deps:
            command.append(str(d.with_suffix(".o")))
            command.append("-I" + str(d.parent))
        for d in std_deps:
            command.append(str(Path(d).with_suffix(".o")))
            command.append("-I" + str(Path(d).parent))
    command.append("-I" + str(base_dir))

    if "gfortran" in compiler and sys.platform == "win32":
        command.append("-static-libgfortran")

    command.append("-o")
    if is_mod:
        command.append(f"{root}.o")
        executable = None
    else:
        executable = test_file.with_suffix("")
        command.append(executable)

    if "module_output_flag" in compiler_info:
        command.append(compiler_info["module_output_flag"])
        command.append(base_dir)

    subprocess.run(command, cwd=path_dir, check=True)

    return executable


# ------------------------------------------------------------------------------
def get_python_output(abs_path, cwd=None):
    assert abs_path.is_absolute()
    p = subprocess.run(
        [sys.executable, abs_path], text=True, capture_output=True, check=True, cwd=cwd
    )
    return p.stdout


# ------------------------------------------------------------------------------
def get_lang_output(abs_path, language):
    if language == "python":
        return get_python_output(abs_path)
    else:
        p = subprocess.run([abs_path], text=True, capture_output=True, check=True)
        return p.stdout


# ------------------------------------------------------------------------------
def get_lang_exit_value(abs_path, language, cwd=None):
    if language == "python":
        if cwd is None:
            p = subprocess.run([sys.executable, abs_path], check=False)
        else:
            p = subprocess.run([sys.executable, abs_path], cwd=cwd, check=False)
    else:
        p = subprocess.run([abs_path], check=False)
    return p.returncode


# ------------------------------------------------------------------------------
def get_value(string, regex, conversion):
    match = regex.search(string)
    assert match
    value = conversion(match.group())
    string = string[match.span()[1] :]
    return value, string


def compare_pyth_fort_output_by_type(p_output, f_output, dtype=float, language=None):

    if dtype is str:
        p_output_split = re.split("\n", p_output)
        f_output_split = re.split("\n", f_output)
        p_list = p_output_split[0].strip()
        f_list = f_output_split[0].strip()
        p_output = "\n".join(p_output_split[1:])
        f_output = "\n".join(f_output_split[1:])
        assert p_list == f_list
    elif dtype is complex:
        rx = re.compile("-?[0-9.]+([eE][+-]?[0-9]+)?j?")
        p, p_output = get_value(p_output, rx, complex)
        if p.imag == 0:
            p2, p_output = get_value(p_output, rx, complex)
            p = p + p2
        if language == "python":
            f, f_output = get_value(f_output, rx, complex)
            if f.imag == 0:
                f2, f_output = get_value(f_output, rx, complex)
                f = f + f2
        else:
            rx = re.compile("-?[0-9.]+([eE][+-]?[0-9]+)?")
            f, f_output = get_value(f_output, rx, float)
            f2, f_output = get_value(f_output, rx, float)
            f = f + f2 * 1j
        assert np.isclose(p, f)
    elif dtype is bool:
        rx = re.compile("TRUE|True|true|1|T|t|FALSE|False|false|F|f|0")
        bool_conversion = lambda m: m.lower() in ["true", "t", "1"]
        p, p_output = get_value(p_output, rx, bool_conversion)
        f, f_output = get_value(f_output, rx, bool_conversion)
        assert p == f

    elif dtype is float:
        rx = re.compile("-?[0-9.]+([eE][+-]?[0-9]+)?")
        p, p_output = get_value(p_output, rx, float)
        f, f_output = get_value(f_output, rx, float)
        assert np.isclose(p, f)

    elif dtype is int:
        rx = re.compile("-?[0-9]+([eE][+-]?[0-9]+)?")
        p, p_output = get_value(p_output, rx, int)
        f, f_output = get_value(f_output, rx, int)
        assert p == f
    else:
        raise NotImplementedError("Type comparison not implemented")
    return p_output, f_output


# ------------------------------------------------------------------------------
def compare_pyth_fort_output(p_output, f_output, dtype=float, language=None):

    if isinstance(dtype, list):
        for d in dtype:
            p_output, f_output = compare_pyth_fort_output_by_type(
                p_output, f_output, d, language=language
            )
    elif dtype is complex:
        while len(p_output) > 0 and len(f_output) > 0:
            p_output, f_output = compare_pyth_fort_output_by_type(
                p_output, f_output, complex, language=language
            )
    elif dtype is str:
        compare_pyth_fort_output_by_type(p_output, f_output, dtype)
    else:
        p_output = p_output.strip().split()
        f_output = f_output.strip().split()
        for p, f in zip(p_output, f_output):
            compare_pyth_fort_output_by_type(p, f, dtype)


# ------------------------------------------------------------------------------
def pyccel_test(
    test_file,
    dependencies=None,
    compile_with_pyccel=True,
    cwd=None,
    pyccel_commands="",
    output_dtype=float,
    language=None,
    *,
    isolated_dir,
):
    """
    Run pyccel and compare the output to ensure that the results
    are equivalent

    Parameters
    ----------
    test_file : str
                The name of the file containing the program, relative
                to the folder containing this file
    dependencies : str/list
                The name of any files which are called by the
                test_file and must therefore be pyccelized in
                order to run it. The paths must be relative to the
                folder containing this file
    compile_with_pyccel : bool
                Indicates whether the compilation step should
                be handled by a basic call to gfortran/gcc (False)
                or internally by pyccel (True)
                default : True
    cwd : str
                The directory from which pyccel and other executables
                will be called, relative to the folder containing
                this file
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
    isolated_dir : Path
                The test-exclusive directory (typically pytest's
                `tmp_path` fixture) into which test files are copied
                and pyccel is run
    """

    if dependencies is None:
        dependencies = []
    elif isinstance(dependencies, str):
        dependencies = [dependencies]

    (test_file,) = copy_to_isolated_dir(isolated_dir, [test_file])
    dependencies = copy_to_isolated_dir(isolated_dir, dependencies)

    test_dir = test_file.parent

    if cwd is None:
        cwd = test_dir
    else:
        assert not Path(cwd).is_absolute(), "cwd must be relative"
        cwd = isolated_dir / cwd

    pyth_out = get_python_output(test_file, cwd)

    if language:
        pyccel_commands += f" --language={language}"
    else:
        language = "fortran"

    output_dir = isolated_dir / "__pyccel__" if language == "python" else None

    generated_dependencies = []
    if dependencies:
        for d in dependencies:
            if output_dir:
                rel_path = d.parent.relative_to(test_dir)
                output = output_dir / rel_path
                pyc_command = pyccel_commands + f" --output={output}"
            else:
                pyc_command = pyccel_commands

            if not compile_with_pyccel:
                dep_output = compile_pyccel(cwd, d, pyc_command + " -t")
                generated_dep = parse_generated_file(
                    dep_output, expected_extensions[language]
                )
                if language == "fortran":
                    compile_fortran(cwd, d, generated_dep, [], is_mod=True)
                elif language == "c":
                    compile_c(cwd, d, generated_dep, [], is_mod=True)
                generated_dependencies.append(generated_dep)
            else:
                compile_pyccel(cwd, d, pyc_command)

    if output_dir:
        pyccel_commands += " --output " + str(output_dir)
        output_test_file = output_dir / test_file.name
    else:
        output_test_file = test_file

    if compile_with_pyccel:
        full_output = compile_pyccel(cwd, test_file, pyccel_commands)
        test_exe = parse_generated_executable(full_output, language)
    else:
        test_output = compile_pyccel(cwd, test_file, pyccel_commands + " -t")
        generated_file = parse_generated_file(
            test_output, expected_extensions[language]
        )
        if language == "fortran":
            test_exe = compile_fortran(
                cwd, output_test_file, generated_file, generated_dependencies
            )
        elif language == "c":
            test_exe = compile_c(
                cwd, output_test_file, generated_file, generated_dependencies
            )
        elif language == "python":
            test_exe = output_test_file
        else:
            raise RuntimeError("Testing unknown language")

    lang_out = get_lang_output(test_exe, language)
    compare_pyth_fort_output(pyth_out, lang_out, output_dtype, language)


# ==============================================================================
# UNIT TESTS
# ==============================================================================
def test_relative_imports_in_project(language, tmp_path):

    dependencies = [
        "project_rel_imports/project/folder1/mod1.py",
        "project_rel_imports/project/folder2/mod2.py",
        "project_rel_imports/project/folder2/mod3.py",
    ]
    pyccel_test(
        "project_rel_imports/runtest.py",
        dependencies,
        cwd="project_rel_imports",
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_absolute_imports_in_project(language, tmp_path):

    dependencies = [
        "project_abs_imports/project/folder1/mod1.py",
        "project_abs_imports/project/folder2/mod2.py",
        "project_abs_imports/project/folder2/mod3.py",
    ]
    pyccel_test(
        "project_abs_imports/runtest.py",
        dependencies,
        cwd="project_abs_imports",
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_rel_imports_python_accessible_folder(language, tmp_path):
    # pyccel is called on scripts/folder2/runtest_rel_imports.py from the scripts folder
    # From this folder python understands relative imports
    from scripts.folder2.runtest_rel_imports import test_func

    pyth_out = str(test_func())

    run_import_path, folder2_funcs_path, runtest_imports_path = copy_to_isolated_dir(
        tmp_path,
        [
            "run_import_function.py",
            "scripts/folder2/folder2_funcs.py",
            "scripts/folder2/runtest_rel_imports.py",
        ],
    )
    path_dir = tmp_path / "scripts"

    pyccel_opt = f"--language={language}"
    if language == "python":
        pyccel_opt += f" --output={tmp_path / '__pyccel__/folder2'}"
    compile_pyccel(
        path_dir / "folder2",
        folder2_funcs_path,
        pyccel_opt,
    )
    compile_pyccel(path_dir, runtest_imports_path, pyccel_opt)
    if language == "python":
        test_location = "__pyccel__.folder2.runtest_rel_imports"
    else:
        test_location = "scripts.folder2.runtest_rel_imports"
    p = subprocess.run(
        [
            sys.executable,
            str(run_import_path),
            test_location,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    fort_out = p.stdout

    compare_pyth_fort_output(pyth_out, fort_out)


# ------------------------------------------------------------------------------
def test_multi_imports_project(language, tmp_path):

    dependencies = [
        "project_multi_imports/file1.py",
        "project_multi_imports/file2.py",
        "project_multi_imports/file3.py",
    ]
    pyccel_test(
        "project_multi_imports/file4.py",
        dependencies,
        cwd="project_multi_imports",
        language=language,
        output_dtype=str,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_imports_compile(language, tmp_path):
    pyccel_test(
        "scripts/runtest_imports.py",
        "scripts/funcs.py",
        compile_with_pyccel=False,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_imports_in_folder(language, tmp_path):
    pyccel_test(
        "scripts/runtest_folder_imports.py",
        "scripts/folder1/folder1_funcs.py",
        compile_with_pyccel=False,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_imports(language, tmp_path):
    pyccel_test(
        "scripts/runtest_imports.py",
        "scripts/funcs.py",
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_folder_imports(language, tmp_path):
    # pyccel is called on scripts/folder2/runtest_imports2.py from the scripts/folder2 folder
    # which is where the final .so file should be
    # From this folder python doesn't understand relative imports
    from scripts.folder2.runtest_imports2 import test_func

    pyth_out = str(test_func())

    run_import_path, folder1_funcs_path, runtest_imports_path = copy_to_isolated_dir(
        tmp_path,
        [
            "run_import_function.py",
            "scripts/folder1/folder1_funcs.py",
            "scripts/folder2/runtest_imports2.py",
        ],
    )
    path_dir = tmp_path / "scripts"

    language_opt = "--language={}".format(language)
    pyccel_opt = language_opt
    if language == "python":
        pyccel_opt = language_opt + f" --output={tmp_path / '__pyccel__' / 'folder1'}"
    compile_pyccel(
        path_dir / "folder1",
        folder1_funcs_path,
        pyccel_opt,
    )
    if language == "python":
        pyccel_opt = language_opt + f" --output={tmp_path / '__pyccel__' / 'folder2'}"
    compile_pyccel(
        path_dir / "folder2",
        runtest_imports_path,
        pyccel_opt,
    )

    if language == "python":
        test_location = "__pyccel__.folder2.runtest_imports2"
    else:
        test_location = "scripts.folder2.runtest_imports2"
    p = subprocess.run(
        [
            sys.executable,
            run_import_path,
            test_location,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    fort_out = p.stdout

    compare_pyth_fort_output(pyth_out, fort_out)


# ------------------------------------------------------------------------------
def test_funcs(language, tmp_path):
    pyccel_test("scripts/runtest_funcs.py", language=language, isolated_dir=tmp_path)


def test_capitalised_language(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/runtest_funcs.py"])
    cwd = test_file.parent
    output_folder = tmp_path / "__pyccel__"
    compile_pyccel(
        cwd, test_file, f"--language={language.capitalize()} --output={output_folder}"
    )


# ------------------------------------------------------------------------------
# Enumerate not supported in c
def test_inout_func(language, tmp_path):
    pyccel_test(
        "scripts/runtest_inoutfunc.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
def test_bool(language, tmp_path):
    pyccel_test(
        "scripts/bool_comp.py",
        output_dtype=bool,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_expressions(experimental_language, tmp_path):
    types = (
        [float, complex, int, float, float, int]
        + [float] * 3
        + [complex, int, complex, complex, int, int, float]
        + [complex] * 3
        + [float] * 3
        + [int]
        + [float] * 2
        + [int]
        + [float] * 3
        + [int]
        + [float] * 3
        + [int] * 2
        + [float] * 2
        + [int] * 5
        + [complex]
        + [bool] * 9
    )
    pyccel_test(
        "scripts/expressions.py",
        language=experimental_language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.fortran
def test_generic_functions(tmp_path):
    # Only testing Fortran for simple compilation outside of Pyccel
    pyccel_test(
        "scripts/runtest_generic_functions.py",
        dependencies="scripts/generic_functions.py",
        compile_with_pyccel=False,
        output_dtype=[
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            float,
            int,
            int,
        ],
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_default_arguments(language, tmp_path):
    pyccel_test(
        "scripts/runtest_default_args.py",
        dependencies="scripts/default_args_mod.py",
        output_dtype=[
            int,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            bool,
            bool,
            bool,
            float,
            float,
            float,
            float,
            int,
            int,
            float,
            float,
            float,
            float,
        ],
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_pyccel_calling_directory(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/runtest_funcs.py"])
    cwd = tmp_path

    pyth_out = get_python_output(test_file)

    language_opt = f"--language={language}"
    output = compile_pyccel(cwd, test_file, language_opt)

    test_exe = parse_generated_executable(output, language)
    fort_out = get_lang_output(test_exe, language)

    compare_pyth_fort_output(pyth_out, fort_out)


# ------------------------------------------------------------------------------
def test_in_specified(language, tmp_path):
    pyccel_test(
        "scripts/runtest_degree_in.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/hope_benchmarks/fib.py",
        "scripts/hope_benchmarks/quicksort.py",
        "scripts/hope_benchmarks/hope_pisum.py",
        "scripts/hope_benchmarks/hope_ln_python.py",
        "scripts/hope_benchmarks/hope_pairwise_python.py",
        "scripts/hope_benchmarks/point_spread_func.py",
        "scripts/hope_benchmarks/simplify.py",
    ],
)
def test_hope_benchmarks(test_file, language, tmp_path):
    pyccel_test(test_file, language=language, isolated_dir=tmp_path)


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/import_syntax/from_mod_import.py",
        "scripts/import_syntax/from_mod_import_as.py",
        "scripts/import_syntax/import_mod.py",
        "scripts/import_syntax/import_mod_as.py",
        "scripts/import_syntax/from_mod_import_func.py",
        "scripts/import_syntax/from_mod_import_as_func.py",
        "scripts/import_syntax/import_mod_func.py",
        "scripts/import_syntax/import_mod_as_func.py",
        "scripts/import_syntax/collisions3.py",
        "scripts/import_syntax/collisions5.py",
    ],
)
def test_import_syntax(test_file, language, tmp_path):
    pyccel_test(test_file, language=language, isolated_dir=tmp_path)


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/import_syntax/from_mod_import_as_user_func.py",
        "scripts/import_syntax/from_mod_import_as_user.py",
        "scripts/import_syntax/collisions2.py",
        "scripts/runtest_import_mod_project_as.py",
    ],
)
def test_import_syntax_user_as(test_file, language, tmp_path):
    pyccel_test(
        test_file,
        dependencies="scripts/import_syntax/user_mod.py",
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/import_syntax/from_mod_import_user.py",
        "scripts/import_syntax/import_mod_user.py",
        "scripts/import_syntax/import_mod_as_user.py",
        "scripts/import_syntax/from_mod_import_user_func.py",
        "scripts/import_syntax/import_mod_user_func.py",
        "scripts/import_syntax/import_mod_as_user_func.py",
    ],
)
def test_import_syntax_user(test_file, language, tmp_path):
    pyccel_test(
        test_file,
        dependencies="scripts/import_syntax/user_mod.py",
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_import_collisions(language, tmp_path):
    pyccel_test(
        "scripts/import_syntax/collisions4.py",
        dependencies=[
            "scripts/import_syntax/user_mod.py",
            "scripts/import_syntax/user_mod2.py",
        ],
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_import_collisions_builtins(language, tmp_path):
    pyccel_test(
        "scripts/import_syntax/collisions6.py",
        dependencies=["scripts/import_syntax/user_mod_builtin_conflict.py"],
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_class_import_as(language, tmp_path):
    pyccel_test(
        "scripts/import_syntax/from_cls_mod_import_as_user.py",
        dependencies=["scripts/import_syntax/user_cls_mod.py"],
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_numpy_kernels_compile(language, tmp_path):
    pyccel_opt = f"--language={language}"
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/numpy/numpy_kernels.py"])
    compile_pyccel(test_file.parent, test_file, pyccel_opt)


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_randint_size_program(language, tmp_path):
    pyccel_test(
        "scripts/numpy/randint_size.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
def test_multiple_results(language, tmp_path):
    pyccel_test(
        "scripts/runtest_multiple_results.py",
        output_dtype=[
            int,
            float,
            complex,
            bool,
            int,
            complex,
            int,
            bool,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            int,
        ],
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_elemental(language, tmp_path):
    pyccel_test(
        "scripts/decorators_elemental.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
def test_print_strings(experimental_language, tmp_path):
    types = str
    pyccel_test(
        "scripts/print_strings.py",
        language=experimental_language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("python", marks=pytest.mark.python),
        pytest.param("c", marks=pytest.mark.c),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="Can't print NaN in Fortran"),
                pytest.mark.fortran,
            ],
        ),
    ),
)
def test_print_nan(language, tmp_path):
    types = str
    pyccel_test(
        "scripts/print_nan.py",
        language=language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_print_integers(language, tmp_path):
    types = str
    pyccel_test(
        "scripts/print_integers.py",
        language=language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_print_sp_and_end(experimental_language, tmp_path):
    types = str
    pyccel_test(
        "scripts/print_sp_and_end.py",
        language=experimental_language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_c_arrays(language, tmp_path):
    types = (
        [int] * 15
        + [float] * 5
        + [int] * 25
        + [float] * 20 * 5
        + [complex] * 3 * 10
        + [complex] * 5
        + [float] * 10
        + [float] * 6
        + [float] * 2 * 3
        + [complex] * 3 * 10
        + [float] * 2 * 3
        + [int] * 3
    )
    pyccel_test(
        "scripts/c_arrays.py",
        language=language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param("python", marks=pytest.mark.python),
        pytest.param(
            "c",
            marks=[
                pytest.mark.xfail(reason="Negative slices are not handled"),
                pytest.mark.c,
            ],
        ),
    ),
)
def test_arrays_view(language, tmp_path):
    types = (
        [int] * 10
        + [int] * 10
        + [int] * 4
        + [int] * 4
        + [int] * 10
        + [int] * 6
        + [int] * 10
        + [int] * 10
        + [int] * 25
        + [int] * 60
    )
    if platform.system() in ("Darwin", "Windows") and language == "fortran":
        # MacOS compiler incorrectly reports
        # Fortran runtime error: Index '4378074096' of dimension 2 of array 'a' outside of expected range (0:2)
        # At line 208 of file /Users/runner/work/pyccel/pyccel/tests/pyccel/scripts/__pyccel__/arrays_view.f90
        # x(0:) => a(1_i64:, merge(3_i64 + v, v, v < 0_i64))
        pyccel_test(
            "scripts/arrays_view.py",
            language=language,
            output_dtype=types,
            pyccel_commands="--no-debug",
            isolated_dir=tmp_path,
        )
    else:
        pyccel_test(
            "scripts/arrays_view.py",
            language=language,
            output_dtype=types,
            isolated_dir=tmp_path,
        )


# ------------------------------------------------------------------------------
def test_return_numpy_arrays(language, tmp_path):
    types = [int] * 4  # 4 ints for a
    types += [int] * 2  # 2 ints for b
    types += [float] * 2  # 2 floats for c
    types += [bool] * 2  # 2 bools for d
    types += [complex] * 2  # 2 complexes for e
    types += [float] * 5  # 5 floats for h
    types += [int] * 5  # 5 ints for g
    types += [int] * 4  # 4 ints for k
    types += [float] * 48  # 48 floats for x
    pyccel_test(
        "scripts/return_numpy_arrays.py",
        language=language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_array_binary_op(language, tmp_path):
    types = [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 4
    types += [int, float, int, int]
    types += [int] * 8
    pyccel_test(
        "scripts/array_binary_operation.py",
        language=language,
        output_dtype=types,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/classes/classes.py",
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
    ],
)
def test_classes(test_file, language, tmp_path):
    pyccel_test(test_file, language=language, isolated_dir=tmp_path)


def test_class_magic(language, tmp_path):
    pyccel_test(
        "scripts/classes/class_magic.py",
        language=language,
        output_dtype=[int] * 6 + [bool] * 2 + [int],
        isolated_dir=tmp_path,
    )


def test_tuples_in_classes(language, tmp_path):
    test_file = "scripts/classes/tuples_in_classes.py"
    pyccel_test(
        test_file,
        language=language,
        output_dtype=[float, float, float, bool, bool],
        isolated_dir=tmp_path,
    )


def test_classes_type_print(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/classes/empty_class.py"])
    cwd = test_file.parent

    pyccel_commands = f" --language={language}"

    if language == "python":
        pyccel_commands += f" --output={cwd / '__pyccel__'}"

    output = compile_pyccel(cwd, test_file, pyccel_commands)
    test_exe = parse_generated_executable(output, language)

    lang_out = get_lang_output(test_exe, language)

    rx = re.compile(r"\bA\b")
    assert rx.search(lang_out)


def test_class_inline_array(language, tmp_path):
    pyccel_test(
        "scripts/classes/class_inline.py",
        dependencies=["scripts/classes/importable.py"],
        language=language,
        output_dtype=float,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_interfaces_in_classes(language, tmp_path):
    pyccel_test(
        "scripts/classes/generic_methods.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Compilation problem. On execution Windows raises: error while loading shared libraries: liblapack.dll: cannot open shared object file: No such file or directory",
)
@pytest.mark.external
@pytest.mark.fortran
def test_lapack(tmp_path):
    # TODO: Uncomment this when dgetri can be expressed with scipy
    # pyccel_test(test_file)

    # TODO: Remove the rest of the function when dgetri can be expressed with scipy
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/lapack_subroutine.py"])

    cwd = test_file.parent

    output = compile_pyccel(cwd, test_file)

    lang_out = get_lang_output(parse_generated_executable(output), "fortran")
    rx = re.compile("[-0-9.eE]+")
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


# ------------------------------------------------------------------------------
def test_type_print(experimental_language, tmp_path):
    pyccel_test(
        "scripts/runtest_type_print.py",
        language=experimental_language,
        output_dtype=str,
        isolated_dir=tmp_path,
    )


def test_container_type_print(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(
        tmp_path, ["scripts/runtest_array_type_print.py"]
    )
    cwd = test_file.parent

    pyccel_commands = f" --language={language}"

    if language == "python":
        pyccel_commands += f" --output={cwd / '__pyccel__'}"

    output = compile_pyccel(cwd, test_file, pyccel_commands)
    test_exe = parse_generated_executable(output, language)

    lang_out = get_lang_output(test_exe, language)

    rx = re.compile(r"\bnumpy.ndarray\b")
    assert rx.search(lang_out)

    if language != "python":
        rx = re.compile(r"\bfloat64\b")
        assert rx.search(lang_out)


# ------------------------------------------------------------------------------


def test_module_init(language, tmp_path):
    test_mod, test_prog = copy_to_isolated_dir(
        tmp_path, ["scripts/module_init.py", "scripts/runtest_module_init.py"]
    )

    pyth_out = get_python_output(test_prog)

    cwd = test_prog.parent

    pyccel_commands = f"--language={language}"

    compile_pyccel(cwd, test_mod, pyccel_commands)

    if language != "python":
        pyth_mod_out = get_python_output(test_prog, cwd)
        compare_pyth_fort_output(pyth_out, pyth_mod_out, str, language)

    prog_output = compile_pyccel(cwd, test_prog, pyccel_commands)
    test_exe = parse_generated_executable(prog_output, language)

    lang_out = get_lang_output(test_exe, language)

    compare_pyth_fort_output(pyth_out, lang_out, str, language)


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/asserts/valid_assert.py",
        "scripts/asserts/invalid_assert1.py",
        "scripts/asserts/invalid_assert2.py",
        "scripts/asserts/invalid_assert3.py",
    ],
)
def test_assert(language, test_file, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, [test_file])

    pyth_out = get_lang_exit_value(test_file, "python")

    output_dir = tmp_path / "__pyccel__"

    cwd = tmp_path

    pyccel_commands = f" --language={language}"
    pyccel_commands += f" --output={output_dir}"
    pyccel_commands += " --debug"

    output = compile_pyccel(cwd, test_file, pyccel_commands)
    test_exe = parse_generated_executable(output, language)
    lang_out = get_lang_exit_value(test_exe, language)
    assert (not lang_out and not pyth_out) or (lang_out and pyth_out)


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "test_file",
    [
        "scripts/exits/empty_exit.py",
        "scripts/exits/negative_exit1.py",
        "scripts/exits/negative_exit2.py",
        "scripts/exits/positive_exit1.py",
        "scripts/exits/positive_exit2.py",
        "scripts/exits/positive_exit3.py",
        "scripts/exits/zero_exit.py",
        "scripts/exits/error_message_exit.py",
    ],
)
def test_exit(language, test_file, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, [test_file])

    pyth_out = get_lang_exit_value(test_file, "python")

    output_dir = tmp_path / "__pyccel__"

    cwd = tmp_path

    if not language:
        language = "fortran"
    pyccel_commands = f" --language={language}"
    pyccel_commands += f" --output={output_dir}"

    output = compile_pyccel(cwd, test_file, pyccel_commands)
    test_exe = parse_generated_executable(output, language)
    lang_out = get_lang_exit_value(test_exe, language)
    assert lang_out == pyth_out


# ------------------------------------------------------------------------------
def test_module_init_collisions(language, tmp_path):
    test_mod, test_prog = copy_to_isolated_dir(
        tmp_path, ["scripts/module_init2.py", "scripts/runtest_module_init2.py"]
    )

    pyth_out = get_python_output(test_prog)

    cwd = tmp_path

    pyccel_commands = f"--language={language}"

    compile_pyccel(cwd, test_mod, pyccel_commands)
    prog_output = compile_pyccel(cwd, test_prog, pyccel_commands)

    lang_out = get_lang_output(
        parse_generated_executable(prog_output, language), language
    )

    compare_pyth_fort_output(
        pyth_out,
        lang_out,
        [float, float, float, int, float, float, float, int],
        language,
    )


@pytest.mark.fortran
def test_function_aliasing(tmp_path):
    pyccel_test(
        "scripts/runtest_function_alias.py", language="fortran", isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------


def test_function(language, tmp_path):
    pyccel_test(
        "scripts/functions.py",
        language=language,
        output_dtype=str,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif_by_language(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="1671",
    language="fortran",
)
def test_inline(language, tmp_path):
    pyccel_test(
        "scripts/decorators_inline.py", language=language, isolated_dir=tmp_path
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif_by_language(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="1671",
    language="fortran",
)
def test_inline_import(language, tmp_path):
    pyccel_test(
        "scripts/runtest_decorators_inline.py",
        dependencies=("scripts/decorators_inline.py"),
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.language_agnostic
def test_json(tmp_path):
    cmd = [
        shutil.which("pyccel"),
        "config",
        "export",
        f"{tmp_path}/test.json",
        "--compiler-family",
        "intel",
    ]
    subprocess.run(cmd, check=True)
    with open(tmp_path / "test.json", "r", encoding="utf-8") as f:
        dict_1 = json.load(f)
    assert dict_1["c"]["exec"] == "icx"
    cmd = [
        shutil.which("pyccel"),
        "config",
        "export",
        f"{tmp_path}/test2.json",
        "--compiler-config",
        f"{tmp_path}/test.json",
    ]
    subprocess.run(cmd, check=True)
    with open(tmp_path / "test2.json", "r", encoding="utf-8") as f:
        dict_2 = json.load(f)

    assert dict_1 == dict_2


@pytest.mark.language_agnostic
def test_json_relative_path(tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/runtest_funcs.py"])

    cmd = [shutil.which("pyccel"), "config", "export", str(tmp_path / "test.json")]
    subprocess.run(cmd, check=True)

    new_dir = tmp_path / "new_dir"
    new_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp_path / "test.json", new_dir / "test.json")
    compile_pyccel(
        new_dir,
        test_file,
        "--compiler-config test.json",
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param("c", marks=pytest.mark.c),
    ),
)
def test_json_register(language, tmp_path):
    current_config_folder = os.environ.get("PYCCEL_CONFIG_HOME", None)
    try:
        example_json_path = str(tmp_path / "test.json")
        cmd = [shutil.which("pyccel"), "config", "export", example_json_path]
        subprocess.run(cmd, check=True)
        with open(example_json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config[language]["debug_flags"] = "-g -O0"

        bad_format_json_path = tmp_path / "bad_format.json"
        with open(bad_format_json_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # Check `pyccel config check` command
        cmd = [shutil.which("pyccel"), "config", "check", str(bad_format_json_path)]
        p = subprocess.run(cmd, check=False)
        assert p.returncode == 1

        config[language]["debug_flags"] = ["-g", "-O0"]
        config[language]["general_flags"] = ["--version"]

        os.environ["PYCCEL_CONFIG_HOME"] = str(tmp_path)

        # Check registration
        timing_json_path = tmp_path / "timing.json"
        with open(timing_json_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        cmd = [
            shutil.which("pyccel"),
            "config",
            "register",
            "compiler_timing",
            str(timing_json_path),
        ]
        subprocess.run(cmd, check=True)

        # Check that existing compiler-family can't be overwritten
        cmd = [
            shutil.which("pyccel"),
            "config",
            "register",
            "compiler_timing",
            example_json_path,
        ]
        p = subprocess.run(cmd, check=False)
        assert p.returncode == 1

        (test_file,) = copy_to_isolated_dir(
            tmp_path, ["scripts/array_binary_operation.py"]
        )

        cmd = [
            shutil.which("pyccel"),
            "compile",
            "--compiler-family=compiler_timing",
            f"--language={language}",
            "-v",
            test_file,
        ]
        p = subprocess.run(cmd, check=True, text=True, capture_output=True)
        received_output = p.stdout

        # Check version output for selected compiler
        cmd = [shutil.which(config[language]["exec"]), "--version"]
        p = subprocess.run(cmd, check=True, text=True, capture_output=True)
        expected_output = p.stdout

        # Check version output is present to ensure new config is being used
        assert expected_output in received_output

        cmd = [shutil.which("pyccel"), "config", "remove", "compiler_timing"]
        subprocess.run(cmd, check=True)

    finally:
        if current_config_folder:
            os.environ["PYCCEL_CONFIG_HOME"] = current_config_folder
        else:
            os.environ.pop("PYCCEL_CONFIG_HOME", None)


# ------------------------------------------------------------------------------
@pytest.mark.language_agnostic
def test_reserved_file_name():
    plugin_manager = pluggy.PluginManager("pyccel")

    with pytest.raises(ValueError) as exc_info:
        libname = str(random.choice(tuple(python_builtin_libs))) + ".py"  # nosec B311
        execute_pyccel(fname=libname, plugin_manager=plugin_manager)
    assert (
        str(exc_info.value)
        == f"File called {libname} has the same name as a Python built-in package and can't be imported from Python. See #1402"
    )


# ------------------------------------------------------------------------------
@pytest.mark.skip(reason="List concatenation not yet implemented")
def test_concatenation(language, tmp_path):
    pyccel_test(
        "scripts/concatenation.py",
        language=language,
        output_dtype=[int] * 15 + [str],
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param("c", marks=pytest.mark.c),
    ),
)
def test_class_imports(language, tmp_path):
    # runtest.py unconditionally imports all three modules, so every source
    # file must already be present (even before it is individually
    # pyccelized) for any of the checks below to run.
    test_file, point_mod, line_mod, square_mod = copy_to_isolated_dir(
        tmp_path,
        [
            "project_class_imports/runtest.py",
            "project_class_imports/project/basics/Point_mod.py",
            "project_class_imports/project/basics/Line_mod.py",
            "project_class_imports/project/shapes/Square_mod.py",
        ],
    )
    cwd = test_file.parent

    pyth_out = get_python_output(test_file, cwd)

    compile_pyccel(cwd, point_mod, f"--language={language} --verbose")

    out1 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out1, float, "python")

    compile_pyccel(cwd, line_mod, f"--language={language} --verbose")

    out2 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out2, float, "python")

    compile_pyccel(cwd, square_mod, f"--language={language} --verbose")

    out3 = get_python_output(test_file, cwd)
    compare_pyth_fort_output(pyth_out, out3, float, "python")

    output = compile_pyccel(cwd, test_file, f"--language={language} --verbose")

    lang_out = get_lang_output(parse_generated_executable(output, language), language)
    compare_pyth_fort_output(pyth_out, lang_out, float, language)


# ------------------------------------------------------------------------------
def test_time_execution_flag(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/runtest_funcs.py"])

    cwd = test_file.parent

    cmd = [
        shutil.which("pyccel"),
        "compile",
        str(test_file),
        f"--language={language}",
        "--time-execution",
    ]
    if language == "python":
        cmd.append(f"--output={cwd/ '__pyccel__'}")
    p = subprocess.run(cmd, text=True, cwd=cwd, capture_output=True, check=True)

    result_lines = p.stdout.split("\n")
    assert "Timers" in result_lines[0]
    assert "Total" in result_lines[-2]
    for l in result_lines[1:-1]:
        assert " : " in l


# ------------------------------------------------------------------------------
def test_module_name_containing_conflict(language, tmp_path):
    endif_file, test_file = copy_to_isolated_dir(
        tmp_path,
        ["scripts/endif.py", "scripts/runtest_badly_named_module.py"],
    )
    compile_pyccel(tmp_path, endif_file, options=f"--language={language}")

    out1 = get_python_output(test_file)
    out2 = get_python_output(test_file)

    assert out1 == out2


# ------------------------------------------------------------------------------
def test_stubs(language, tmp_path):
    """
    This tests that a stub file is generated and ensures the stub files are
    still generated with the expected format. However it is not a good test.
    It prevents any changes being made to the output format and doesn't
    check that it can be parsed. This test should be replaced once stub files
    can be read.
    """
    with open(
        Path(__file__).resolve().parent / f"scripts/runtest_stub.{language}.pyi",
        "r",
        encoding="utf-8",
    ) as f:
        expected_pyi = f.read()

    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/runtest_stub.py"])

    pyc_output = compile_pyccel(
        tmp_path,
        test_file,
        options=f"--language={language} -t",
    )
    with open(
        parse_generated_file(pyc_output, ".pyi"),
        "r",
        encoding="utf-8",
    ) as f:
        generated_pyi = f.read()

    assert expected_pyi == generated_pyi


# ------------------------------------------------------------------------------
def test_builtin_container_print(language, tmp_path):
    pyccel_test(
        "scripts/print_builtin_containers.py",
        output_dtype=str,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_pyccel_generated_compilation_dependency(language, tmp_path):
    pyccel_test(
        "scripts/runtest_pyccel_generated_compilation_dependency.py",
        dependencies=["scripts/pyccel_generated_compilation_dependency.py"],
        output_dtype=int,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_generated_name_collision(language, tmp_path):
    pyccel_test(
        "scripts/GENERATED_NAME_COLLISION.py",
        output_dtype=int,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_array_tuple_shape(language, tmp_path):
    pyccel_test(
        "scripts/array_tuple_shape.py",
        output_dtype=int,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_varargs(language, tmp_path):
    pyccel_test("scripts/runtest_varargs.py", language=language, isolated_dir=tmp_path)


# ------------------------------------------------------------------------------
@pytest.mark.python
def test_varkwargs(tmp_path):
    pyccel_test(
        "scripts/runtest_varkwargs.py",
        language="python",
        output_dtype=str,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif_by_language(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="1671",
    language="fortran",
)
def test_inline_using_import(language, tmp_path):
    pyccel_test(
        "scripts/inlining/runtest_inline_using_import.py",
        dependencies=[
            "scripts/inlining/my_func.py",
            "scripts/inlining/my_other_func.py",
            "scripts/inlining/inline_using_import.py",
        ],
        language=language,
        output_dtype=float,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif_by_language(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="1671",
    language="fortran",
)
def test_inline_using_import_2(language, tmp_path):
    pyccel_test(
        "scripts/inlining/runtest_inline_using_import_2.py",
        dependencies=[
            "scripts/inlining/my_func.py",
            "scripts/inlining/my_other_func.py",
            "scripts/inlining/inline_using_import.py",
        ],
        language=language,
        output_dtype=float,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
@pytest.mark.skipif_by_language(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="1671",
    language="fortran",
)
def test_inline_using_named_import(language, tmp_path):
    pyccel_test(
        "scripts/inlining/runtest_inline_using_named_import.py",
        dependencies=[
            "scripts/inlining/my_func.py",
            "scripts/inlining/my_func2.py",
            "scripts/inlining/inline_using_named_import.py",
        ],
        language=language,
        output_dtype=float,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_classes_array_property(language, tmp_path):
    pyccel_test(
        "scripts/classes/runtest_classes_array_property.py",
        dependencies=["scripts/classes/classes_array_property.py"],
        language=language,
        output_dtype=float,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_classes_pointer_import(language, tmp_path):
    test_file, dependency = copy_to_isolated_dir(
        tmp_path,
        [
            "scripts/classes/runtest_class_pointer_2.py",
            "scripts/classes/class_pointer_2.py",
        ],
    )
    cwd = test_file.parent

    pyth_out = get_python_output(test_file, cwd)

    compile_pyccel(cwd, dependency, f"--language={language}")

    pyth_interface_out = get_python_output(test_file, cwd)
    assert pyth_out == pyth_interface_out

    output = compile_pyccel(cwd, test_file, f"--language={language}")

    lang_out = get_lang_output(parse_generated_executable(output, language), language)
    compare_pyth_fort_output(pyth_out, lang_out, float, language)


# ------------------------------------------------------------------------------
def test_functional_statements(language, tmp_path):
    pyccel_test(
        "scripts/functional_statements.py",
        output_dtype=[int] * 9,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_complex_numbers(language, tmp_path):
    pyccel_test(
        "scripts/complex_numbers.py",
        output_dtype=[complex] * 6,
        language=language,
        isolated_dir=tmp_path,
    )


# ------------------------------------------------------------------------------
def test_line_annotation_plugin(language, tmp_path):
    (test_file,) = copy_to_isolated_dir(tmp_path, ["scripts/funcs.py"])
    folder = test_file.parent

    # Choose an output folder that cannot contain translations of other files
    output_folder = folder / "__pyccel__la__"

    pyccel_commands = (
        f"-t --line_annotation --language={language} --output={output_folder}"
    )

    shutil.rmtree(output_folder, ignore_errors=True)

    compile_pyccel(folder, test_file, pyccel_commands)

    for fi in output_folder.iterdir():
        if fi.suffix in (".c", ".f90", ".py"):
            with open(fi, "r", encoding="utf-8") as file:
                code = file.read()
            assert str(test_file) in code
