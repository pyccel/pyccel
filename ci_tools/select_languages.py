"""Determine which pyccel target languages need to be tested for a given diff.

Compares the files changed between a base commit/branch and HEAD against the
known paths of each language's printer/wrapper. If every changed file belongs
exclusively to one language, only that language is selected. If any changed
file falls outside of these narrow, well-known paths (e.g. shared codegen
stages, the AST, the parser, the stdlib, tests, or CI configuration) every
language is selected, since such a file could affect any backend.
"""

import argparse
import os
import subprocess

FORTRAN_PATHS = (
    "pyccel/codegen/printing/fcode.py",
    "pyccel/codegen/wrapper/fortran_to_c_wrapper.py",
)
C_PATHS = (
    "pyccel/codegen/printing/ccode.py",
    "pyccel/codegen/printing/cwrappercode.py",
)
CPP_PATHS = (
    "pyccel/codegen/printing/cppcode.py",
    "pyccel/codegen/wrapper/cpp_to_python_wrapper.py",
    "pyccel/codegen/printing/pybindcode.py",
)
PYTHON_PATHS = ("pyccel/codegen/printing/pycode.py",)

LANGUAGE_PATHS = {
    "fortran": FORTRAN_PATHS,
    "c": C_PATHS,
    "cpp": CPP_PATHS,
    "python": PYTHON_PATHS,
}


def get_changed_files(base):
    """
    Get the list of files changed between `base` and HEAD.

    Returns None if the diff could not be computed (e.g. the base ref is
    unavailable), so that callers can fail safe and run every language.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def select_languages(changed_files):
    """
    Decide which languages need testing given a list of changed files.

    Returns a dict mapping each language to a bool indicating whether it
    needs to be tested. Any file that isn't recognised as belonging
    exclusively to one language's printer/wrapper causes every language to
    be selected (safe default).
    """
    if changed_files is None:
        return dict.fromkeys(LANGUAGE_PATHS, True)

    run = dict.fromkeys(LANGUAGE_PATHS, False)

    for f in changed_files:
        matched_language = next(
            (lang for lang, paths in LANGUAGE_PATHS.items() if f in paths), None
        )
        if matched_language is None:
            return dict.fromkeys(LANGUAGE_PATHS, True)
        run[matched_language] = True

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, help="Base commit/branch to diff against"
    )
    args = parser.parse_args()

    changed_files = get_changed_files(args.base)
    run = select_languages(changed_files)

    print("Changed files:", changed_files if changed_files is not None else "<unknown>")
    for lang, should_run in run.items():
        print(f"run_{lang}={str(should_run).lower()}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            for lang, should_run in run.items():
                f.write(f"run_{lang}={str(should_run).lower()}\n")
            f.write("run_agnostic=true\n")


if __name__ == "__main__":
    main()
