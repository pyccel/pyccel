# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from pyccel.ast.variable import Variable
from pyccel.codegen.printing.ccudacode import CcudaCodePrinter
from pyccel.ast.core      import Deallocate

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_print_deallocate(language):
    # Test case 1: variable is in host memory
    var = Variable('int', 'y', memory_location='host')
    expr = Deallocate(var)
    printer = CcudaCodePrinter('')
    output = printer.doprint(expr)
    expected_output = "cuda_free_host(y);\n"
    assert output == expected_output
    # Test case 2: variable is in device memory
    var = Variable('int', 'y', memory_location='device')
    expr = Deallocate(var)
    printer = CcudaCodePrinter('')
    output = printer.doprint(expr)
    expected_output = "cuda_free(y);\n"
    assert output == expected_output
    #Test case 3: Memory location of variable is managed
    var = Variable('int', 'y', memory_location='managed')
    expr = Deallocate(var)
    printer = CcudaCodePrinter('')
    output = printer.doprint(expr)
    expected_output = "cuda_free(y);\n"
    assert output == expected_output
    #Test case 4: variable is an alias
    var = Variable('int', 'y', memory_handling='alias')
    expr = Deallocate(var)
    printer = CcudaCodePrinter('')
    output = printer.doprint(expr)
    expected_output = "cuda_free_pointer((*y));\n"
    assert output == expected_output
