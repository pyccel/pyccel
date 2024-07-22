# pylint: disable=missing-function-docstring, missing-module-docstring
import subprocess
import os
import pathlib
import sys
import shutil
import pytest

stc_containers = ('list', 'set', 'dict')

def pytest_collect_file(parent, file_path):
    """
    A hook to collect test_*.c test files.

    Parameters
    ----------
    parent : Collector
        The node where the file will be collected.
    file_path : pathlib.PosixPath
        The path to the file which may or may not be collected.
    """
    if file_path.suffix == ".py" and file_path.stem.endswith("_check"):
        return ValgrindTestFile.from_parent(path=pathlib.Path(file_path), parent=parent)
    return None

class ValgrindTestFile(pytest.File):
    """
    A custom file handler class for C unit test files.
    """
    def collect(self):
        """
        The method which collects the test result.

        Overridden collect method to collect the results from each
        C unit test executable.
        """
        for language in ('c','fortran'):
            name = f'{self.path}[{language}]'
            test_item = ValgrindTestItem.from_parent(name = name, parent = self)
            test_item.path = self.path
            test_item.language = language
            yield test_item


class ValgrindTestItem(pytest.Item):

    def runtest(self):
        if self.language == 'fortran' and self.path.stem.split('_')[0] in stc_containers:
            raise pytest.skip.Exception("Containers not yet implemented in Fortran")
        if self.language == 'c' and self.path.stem == 'ndarrays_check':
            raise pytest.skip.Exception("The C translation fails. See #1947")

        # Run the exe that corresponds to the .c file and capture the output.
        test_exe = os.path.splitext(str(self.path))[0]
        test_exe = os.path.relpath(test_exe)
        p = subprocess.run([shutil.which("pyccel"), self.path, "--language", self.language],
                        capture_output = True, universal_newlines = True)
        if p.returncode:
            self._failure_code = p.stderr
            raise ValgrindTestException(self, self.name)

        if sys.platform.startswith("win"):
            test_exe += ".exe"
        p = subprocess.run([shutil.which("valgrind"), "--leak-check=full",
                                "--error-exitcode=1", f"./{test_exe}"],
                            capture_output = True, universal_newlines = True)

        if p.returncode:
            self._failure_code = p.stderr
            raise ValgrindTestException(self, self.name)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, ValgrindTestException):
            return self._failure_code
        return super().repr_failure(excinfo, style=style)

    @property
    def description(self):
        return ""

    def reportinfo(self):
        return self.path, None, self.name

class ValgrindTestException(Exception):
    """Custom exception to distinguish C unit test failures from others."""
