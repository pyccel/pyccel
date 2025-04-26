#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing scripts to run the unit tests of Pyccel
"""
import os
#import subprocess
from argparse import ArgumentParser

install_msg = """
In order to run the tests, Pyccel must be installed with the optional [test] dependencies.
You can do this by running one of the following commands:
  1. Standard install, from PyPI   : pip install "pyccel[test]"
  2. Standard install, from sources: pip install ".[test]"
  3. Editable install, from sources: pip install ".[test]" --editable
"""

def pyccel_test():
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

    """

    import sys

    # Pyccel must be installed
    try:
        import pyccel
    except ImportError:
        print()
        print('ERROR: It appears that Pyccel is not installed.')
        print(install_msg)
        sys.exit(1)

    # Pytest must be installed
    try:
        import pytest
    except ImportError:
        print()
        print('ERROR: It appears that pytest is not installed.')
        print(install_msg)
        sys.exit(1)

    # Find the path to the pyccel module
    # TODO: verify, improve
#    pyccel_path = os.path.dirname(os.path.abspath(pyccel.__file__))

#    subprocess.run(['curl', '-JLO', 'https://github.com/pyccel/pyccel/archive/refs/heads/devel.zip'])
#    subprocess.run(['unzip', '-o', 'pyccel-devel.zip'])

    # TODO: Determine the version of Pyccel that we are using

    # Download the test files
    # TODO: use the correct version of the test files
    from urllib.request import urlretrieve
    print("Downloading the test files from GitHub...")
    zip_url  = 'https://github.com/pyccel/pyccel/archive/refs/heads/devel.zip'
    zip_name = 'pyccel.zip'
    zip_path, _ = urlretrieve(zip_url, filename=zip_name)

    # Unzip the test files
    import zipfile
    print("Unzipping the test files...")
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for file in archive.namelist():
            if file.startswith('pyccel-devel/tests'):
                archive.extract(file, path='.')

    # Change into the test directory
    print("Changing into the test directory...")
    os.chdir('pyccel-devel/tests')

    print("Run the single-process tests which must be run one at a time...")
    retcode = pytest.main(['-ra', '-m (not parallel and xdist_incompatible)'])
    print(f"refcode = {retcode}")

    print("Run the single-process tests which can be run in parallel...")
    retcode = pytest.main(['-ra', '-m (not parallel and not xdist_incompatible)', '-n', 'auto'])
    print(f"refcode = {retcode}")

    # TODO: run the parallel tests
    return retcode


def pyccel_test_command():
    """
    Command line wrapper around the pyccel_test function.

    A wrapper around the pyccel_test function which allows
    command line arguments to be passed to the function.
    """
    parser = ArgumentParser(description='Tool for running the test suite of Pyccel')

    parser.parse_args()

    return pyccel_test()
