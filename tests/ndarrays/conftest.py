import subprocess
import os
import sys
import pytest
import pyccel

def pytest_collect_file(parent, path):
    """
    A hook into py.test to collect test_*.c test files.

    """
    if path.ext == ".c" and path.basename.startswith("test_"):
        return CTestFile.from_parent(parent, path)

def pytest_collection_modifyitems(items):
    for item in items:
        if item.fspath.ext == ".c":
            item._nodeid = item.nodeid + " < " + item.test_result["DSCR"] + " >"

class CTestFile(pytest.File):
    """
    A custom file handler class for C unit test files.

    """
    @classmethod
    def from_parent(cls, parent, fspath):
        return super().from_parent(parent=parent, fspath=fspath)

    
    def collect(self):
        """
        Overridden collect method to collect the results from each
        C unit test executable.

        """
        # Run the exe that corresponds to the .c file and capture the output.
        test_exe = os.path.splitext(str(self.fspath))[0]
        rootdir = str(self.config.rootdir)
        rel_path = os.path.relpath(test_exe)
        if  sys.platform.startswith("win"):
            ndarray_path =  rootdir + "\\pyccel\\stdlib\\ndarrays\\"
        else :
            ndarray_path = rootdir + "/pyccel/stdlib/ndarrays/"
        comp_cmd = "gcc "+ rel_path +".c " + ndarray_path + "*.c -I " + ndarray_path + " -o " + rel_path
        subprocess.run(comp_cmd, shell = 'TRUE')
        if sys.platform.startswith("win"):
                rel_path += ".exe"
        
        test_output = subprocess.check_output("./" + rel_path)

        # Clean up the unit test output and remove non test data lines.
        lines = test_output.decode().split("\n")
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line.startswith("[")]

        # Extract the test metadata from the unit test output.
        test_results = []
        for line in lines:
            token, data = line.split(" ", 1)
            token = token[1:-1]
            # print(token, data)
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
            yield CTestItem.from_parent(test_result["function_name"], self, test_result)


class CTestItem(pytest.Item):
    """
    Pytest.Item subclass to handle each test result item. There may be
    more than one test result from a test function.

    """

    def __init__(self, name, parent, test_result):
        """Overridden constructor to pass test results dict."""
        super().__init__(name, parent)
        self.test_result = test_result
    
    @classmethod
    def from_parent(cls, name, parent, test_result):
        return super().from_parent(name=name, parent=parent, test_result=test_result)
    
    def runtest(self):
        """The test has already been run. We just evaluate the result."""
        # print(self.)
        if self.test_result["condition"] == "FAIL":
            raise CTestException(self, self.name)

        
    def reportinfo(self):
        """"Called to display header information about the test case."""
        return self.fspath, self.test_result["line_number"] - 1 , self.name

    def repr_failure(self, exception):
        """
        Called when runtest() raises an exception. The method is used
        to format the output of the failed test result.

        """
        if isinstance(exception.value, CTestException):
            return ("Test failed : {file_name}:{line_number}\nINFO : {INFO}".format(**self.test_result))



class CTestException(Exception):
    """Custom exception to distinguish C unit test failures from others."""
    pass
