#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing scripts to run the unit tests of Pyccel
"""
import json
import os
import pathlib
import sys
from argparse import ArgumentParser
from importlib.metadata import Distribution

from .argparse_helpers import add_version_flag

PYCCEL_TEST_DESCR = 'Run tests to check installation.'

install_msg = """
In order to run the tests, Pyccel must be installed with the optional [test] dependencies.
You can do this by running one of the following commands:
  1. Standard install, from PyPI   : pip install "pyccel[test]"
  2. Standard install, from sources: pip install ".[test]"
  3. Editable install, from sources: pip install ".[test]" --editable
"""

def pyccel_test(*, folder, dry_run, verbose, language, run_mpi):
    """
    Run the unit tests of Pyccel.

    This function runs the unit tests of Pyccel using pytest.
    If pytest is not installed, it will print an error message
    and exit. It also downloads the test files from GitHub
    and unzips them into the current directory. The function
    then changes into the test directory and runs the tests
    using pytest. The tests are run in two stages: first, the
    single-process tests which must be run one at a time
    are run, and then the single-process tests which can be
    run in parallel are run. The function returns the return
    code of the pytest command.

    If the user stops the tests with Ctrl+C, the function will
    print a message and exit gracefully.

    This function does not provide default values for its
    parameters, because these have to be handled by the command
    line parser.

    Parameters
    ----------
    folder : pathlib.Path or None
        The local folder where the tests are located. If None, the
        function will first try to extract the tests from a local editable
        installation, then download them from the official GitHub repository.
    dry_run : bool
        If True, the function will not run the tests, but will
        print the commands that would be run.
    verbose : int
        The verbosity level of the output. The higher the number,
        the more detailed the output will be.
    language : str
        The target language Pyccel is translating to. Default is 'All'.
    run_mpi : bool
        If True, the function will not run the parallel tests.

    Returns
    -------
    pytest.ExitCode
        The return code of the pytest command. If the tests were
        interrupted by the user, the return code will be
        pytest.ExitCode.INTERRUPTED. If the tests failed, the
        return code will be pytest.ExitCode.TESTS_FAILED.
        If the tests passed, the return code will be
        pytest.ExitCode.OK.
    """

    assert isinstance(folder, (pathlib.Path, type(None)))
    assert isinstance(dry_run, bool)
    assert isinstance(verbose, int)

    # Pyccel must be installed
    try:
        import pyccel
    except ImportError:
        print('ERROR: It appears that Pyccel is not installed.')
        print(install_msg)
        sys.exit(1)

    # Pytest must be installed
    try:
        import pytest
    except ImportError:
        print('ERROR: It appears that pytest is not installed.')
        print(install_msg)
        sys.exit(1)

    # If a folder is provided, use it as the test directory
    if folder is not None:
        test_dir = folder.resolve()
        if not test_dir.is_dir():
            print(f"ERROR: The provided folder '{test_dir}' does not exist or is not a directory.")
            sys.exit(1)
        print(f"Using the provided folder as the test directory: {test_dir}")
    else:
        test_dir = None

    version = pyccel.__version__
    zip_url = f'https://github.com/pyccel/pyccel/archive/refs/tags/v{version}.zip'
    download_location = f'pyccel-v{version}'

    if test_dir is None:
        # Determine the version of Pyccel that we are using
        direct_url = Distribution.from_name("pyccel").read_text("direct_url.json")

        # If a direct URL is provided, use it to determine the test directory
        # Otherwise, download the test files from GitHub
        if direct_url:
            url = json.loads(direct_url)["url"]
            if url.startswith("file://"):
                test_dir = pathlib.Path(url.removeprefix("file://")) / "tests"
                if test_dir.exists():
                    print(f"Using the local test directory from direct URL: {test_dir}")
                else:
                    print(f"Pyccel was installed from source but the source directory {test_dir} has been deleted.")
                    print(("Tests will be downloaded from the devel branch but this may lead to "
                           "failures if the branch does not match or has been updated since the "
                           "last installation"))

                    zip_url = 'https://github.com/pyccel/pyccel/archive/refs/heads/devel.zip'
                    download_location = "Pyccel's devel branch"

    if test_dir is None:

        # Download the test files
        from urllib.request import urlopen
        print(f"Downloading the test files from GitHub for {download_location}...")
        zip_path = 'pyccel.zip'
        with urlopen(zip_url, timeout=5) as response: # nosec urllib_urlopen
            with open(zip_path, 'wb') as output_file:
                output_file.write(response.read())

        test_dir = f'pyccel-{version}/tests'

        # Unzip the test files
        import zipfile
        print("Unzipping the test files...")
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for file in archive.namelist():
                if file.startswith(test_dir):
                    archive.extract(file, path='.')

    # Change into the test directory
    print("Changing into the test directory...")
    print(f"> cd {test_dir}")
    os.chdir(test_dir)

    # Descriptions of the tests:
    desc_1 = "Run the single-process tests which must be run one at a time... [all languages]"
    desc_2 = "Run the single-process tests which can be run in parallel... [language: C]"
    desc_3 = "Run the single-process tests which can be run in parallel... [language: Fortran]"
    desc_4 = "Run the single-process tests which can be run in parallel... [language: Python]"
    descriptions = [desc_1, desc_2, desc_3, desc_4]

    # Commands to run the tests:
    cmd_1 = ['-ra', '-m (xdist_incompatible)']
    cmd_3 = ['-ra', '-m (not xdist_incompatible and language_agnostic)', '-n', 'auto']
    cmd_2 = ['-ra', '-m (not xdist_incompatible and c)', '-n', 'auto']
    cmd_3 = ['-ra', '-m (not xdist_incompatible and fortran)', '-n', 'auto']
    cmd_4 = ['-ra', '-m (not xdist_incompatible and python)', '-n', 'auto']
    commands = [cmd_1, cmd_2, cmd_3, cmd_4]

    if language != 'All':
        cmd_1[-1] = cmd_1[-1].removesuffix(')') + f' and {language})'
        relevant_language = [True,
                *[language == desc.split('language: ')[1].removesuffix(']').lower() for desc in descriptions[1:]]]
        descriptions = [desc for i, desc in enumerate(descriptions) if relevant_language[i]]
        commands = [cmd for i, cmd in enumerate(commands) if relevant_language[i]]

    if verbose > 0:
        verbose_flag = '-' + 'v' * verbose
        for cmd in commands:
            cmd.append(verbose_flag)

    # Set the return code to OK by default
    retcode = pytest.ExitCode.OK

    if run_mpi:
        # Run the parallel tests
        import subprocess

        desc_mpi = "Run the parallel tests... [all languages]"
        cmd_mpi = ['mpirun', '-n', '4', '--oversubscribe', 'pytest', '--with-mpi', '-ra', 'epyccel/test_parallel_epyccel.py']
        if verbose:
            cmd_mpi += ['-' + 'v' * verbose]
        if language != 'All':
            cmd_mpi.append(f'-m={language.lower()}')
        print()
        print(desc_mpi)
        print(f'> {" ".join(cmd_mpi)}')
        if dry_run:
            print("Dry run, not executing the parallel tests.")
            retcode = pytest.ExitCode.OK
        else:
            p = subprocess.run(cmd_mpi, check=False, capture_output=True, universal_newlines=True)
            print(p.stdout)
            if p.returncode != 0:
                print(f"Error running parallel tests. Failed with error code {p.returncode}")
                print(p.stderr)
                retcode = pytest.ExitCode.TESTS_FAILED

    # Run the tests in the specified order
    for desc, cmd in zip(descriptions, commands):
        print()
        print(desc)
        print(f'> pytest {" ".join(cmd)}')
        if dry_run:
            print("Dry run, not executing the tests.")
            retcode = pytest.ExitCode.OK
        else:
            retcode = pytest.main(cmd)
            print(f"\nPytest return code: {retcode.name}")
            if retcode == pytest.ExitCode.INTERRUPTED:
                print("\nTest execution was interrupted by the user, exiting...\n")
                return retcode

    # Return the final return code
    return retcode

def setup_pyccel_test_parser(parser):
    """
    Add the pyccel-test arguments to the parser.

    Add the pyccel-test arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    parser.add_argument('--dry-run', action='store_true',
        help='Run all steps without actually running the tests.')

    parser.add_argument('-v', '--verbose', action='count', default=0,
        help='Increase output verbosity (use -v, -vv for more detailed output).')

    parser.add_argument('--folder', type=pathlib.Path, default=None,
        help="Run tests located in custom folder (default: use Pyccel's distribution).")

    parser.add_argument('--language', choices=('Fortran', 'C', 'Python', 'All'), default='All',
        help='Target language for translation, i.e. the main language of the generated code (default: All).',
        type=str.lower)

    parser.add_argument('--no-mpi', action='store_false', dest='run_mpi',
        help="Do not run the parallel tests.")

def pyccel_test_command():
    """
    Command line wrapper around the pyccel_test function.

    A wrapper around the pyccel_test function which allows
    command line arguments to be passed to the function.

    Returns
    -------
    pytest.ExitCode
        The pytest return code.
    """
    parser = ArgumentParser(description='Tool for running the test suite of Pyccel', add_help = True)

    #... Help and Version
    add_version_flag(parser)

    setup_pyccel_test_parser(parser)

    # Parse the command line arguments
    args = parser.parse_args()

    print("warning: The pyccel-test command is deprecated and will be removed in v2.3. Please use pyccel test.", file=sys.stderr)

    print()
    retcode = pyccel_test(**vars(args))
    print()

    return retcode

