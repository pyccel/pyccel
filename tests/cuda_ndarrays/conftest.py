# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=arguments-differ, inconsistent-return-statements, protected-access, abstract-method/
import subprocess
import os
import pathlib
import sys
import shutil
import pytest

NEEDS_FROM_PARENT = hasattr(pytest.Item, "from_parent")

def pytest_collect_file(parent, path):
    """
    A hook to collect test_*.cu test files.

    """
    if path.ext == ".cu" and path.basename.startswith("test_"):
        if NEEDS_FROM_PARENT:
            return CTestFile.from_parent(path=pathlib.Path(path), parent=parent)
        return CTestFile(parent=parent, path=pathlib.Path(path))

def pytest_collection_modifyitems(items):
    """
    a hook to modify the items before the tests for C Cuda unit test files.

    """
    for item in items:
        if item.fspath.ext == ".cu":
            item._nodeid = item.nodeid + " < " + item.test_result["DSCR"] + " >"

class CTestFile(pytest.File):
    """
    A custom file handler class for C unit test files.

    """

    @classmethod
    def from_parent(cls, **kwargs):
        return super().from_parent(**kwargs)

    def collect(self):
        """
        Overridden collect method to collect the results from each
        C unit test executable.

        """
        # Run the exe that corresponds to the .c file and capture the output.
        test_exe = os.path.splitext(str(self.fspath))[0]
        rootdir = str(self.config.rootdir)
        test_exe = os.path.relpath(test_exe)
        ndarray_path =  os.path.join(rootdir , "pyccel", "stdlib", "ndarrays")
        cuda_ndarray_path = os.path.join(rootdir, "pyccel", "stdlib", "cuda_ndarrays")
        comp_cmd = [shutil.which("nvcc"), test_exe + ".cu",
                    os.path.join(ndarray_path, "ndarrays.c"),
                    os.path.join(cuda_ndarray_path, "cuda_ndarrays.cu"),
                    "-I", ndarray_path,
                    "-I", cuda_ndarray_path,
                    "-o", test_exe,]

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
        for test_result in test_results:
            if NEEDS_FROM_PARENT:
                yield CTestItem.from_parent(name = test_result["function_name"],
                        parent = self, test_result = test_result)
            else:
                yield CTestItem(name = test_result["function_name"], parent = self,
                        test_result = test_result)


class CTestItem(pytest.Item):
    """
    Pytest.Item subclass to handle each test result item. There may be
    more than one test result from a test function.

    """

    def __init__(self, *, test_result, **kwargs):
        """Overridden constructor to pass test results dict."""
        super().__init__(**kwargs)
        self.test_result = test_result

    @classmethod
    def from_parent(cls, *, test_result, **kwargs):
        return super().from_parent(test_result=test_result, **kwargs)

    def runtest(self):
        """The test has already been run. We just evaluate the result."""
        if self.test_result["condition"] == "FAIL":
            raise CTestException(self, self.name)

    def reportinfo(self):
        """"Called to display header information about the test case."""
        return self.fspath, self.test_result["line_number"], self.name

    def repr_failure(self, exception):
        """
        Called when runtest() raises an exception. The method is used
        to format the output of the failed test result.

        """
        if isinstance(exception.value, CTestException):
            return (f"Test failed : {self.test_result['file_name']}:{self.test_result['line_number']} {self.test_result['function_name']} < {self.test_result['DSCR']} >\n INFO : {self.test_result['INFO']}")



class CTestException(Exception):
    """Custom exception to distinguish C unit test failures from others."""
