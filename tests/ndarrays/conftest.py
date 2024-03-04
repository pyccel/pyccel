# pylint: disable=missing-function-docstring, missing-module-docstring
import subprocess
import os
import pathlib
import sys
import shutil
import pytest

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
    if file_path.suffix == ".c" and file_path.name.startswith("test"):
        return CTestFile.from_parent(path=pathlib.Path(file_path), parent=parent)
    return None

class CTestFile(pytest.File):
    """
    A custom file handler class for C unit test files.
    """
    def collect(self):
        """
        The method which collects the test result.

        Overridden collect method to collect the results from each
        C unit test executable.
        """
        # Run the exe that corresponds to the .c file and capture the output.
        test_exe = os.path.splitext(str(self.path))[0]
        rootdir = str(self.config.rootdir)
        test_exe = os.path.relpath(test_exe)
        ndarray_path =  os.path.join(rootdir , "pyccel", "stdlib", "ndarrays")
        comp_cmd = [shutil.which("gcc"), test_exe + ".c",
                    os.path.join(ndarray_path,"ndarrays.c"), "-I", ndarray_path, "-o", test_exe]
        subprocess.run(comp_cmd, check= 'TRUE')
        if sys.platform.startswith("win"):
            test_exe += ".exe"
        test_output = subprocess.check_output("./" + test_exe)

        # Clean up the unit test output and remove non test data lines.
        lines = test_output.decode().split("\n")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line.startswith("[")]

        # Extract the test metadata from the unit test output.
        test_results = []
        for line in lines:
            token, data = line.split(" ", 1)
            token = token[1:-1]
            if token in ("PASS", "FAIL"):
                file_name, line_number, function_name = data.split(":")
                test_results.append({"condition": token,
                                     "file_name": file_name,
                                     "function_name": function_name,
                                     "line_number": int(line_number),
                                     "INFO" : "no data found",
                                     "DSCR" : ""
                                     })
            elif token in ("INFO", "DSCR"):
                test_results[-1][token] = data
        for test_results in test_results:
            yield CTestItem.from_parent(name = test_results["function_name"],
                    parent = self, **test_results)


class CTestItem(pytest.Item):
    """
    Pytest.Item subclass to handle each test result item. There may be
    more than one test result from a test function.

    Parameters
    ----------
    file_name : str
        The file where the test was located.

    line_number : int
        The line where the test is found.

    function_name : str
        The name of the function which was run.

    condition : str
        The condition of the test [PASS/FAIL].

    DSCR : str, optional
        A textual description of the test.

    INFO : str, optional
        Information about the assertion.

    **kwargs : dict
        See pytest.Item for details.
    """

    def __init__(self, *, file_name, line_number, function_name, condition,
            DSCR, INFO, **kwargs):
        super().__init__(**kwargs)
        self._file_name = file_name
        self._line_number = line_number
        self._function_name = function_name
        self._description = DSCR
        self._info = INFO
        self._condition = condition
        self._nodeid = self._nodeid + " < " + self._description + " >"

    def runtest(self):
        """The test has already been run. We just evaluate the result."""
        if self._condition == "FAIL":
            raise CTestException(self, self.name)

    def reportinfo(self):
        """"Called to display header information about the test case."""
        return self.path, self._line_number, self.name

    @property
    def description(self):
        return self._description

    def repr_failure(self, excinfo, style=None):
        """
        Get the error description.

        Called when runtest() raises an exception. The method is used
        to format the output of the failed test result.

        Parameters
        ----------
        excinfo : Exception
            The exception that was raised by the test.
        style : None
            The style of the traceback.
        """
        if isinstance(excinfo.value, CTestException):
            return (f"Test failed : {self._file_name}:{self._line_number} {self._function_name} < {self._description} >\n"
                    f"INFO : {self._info}")
        return super().repr_failure(excinfo, style=style)



class CTestException(Exception):
    """Custom exception to distinguish C unit test failures from others."""
