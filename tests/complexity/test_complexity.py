# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.complexity.memory import MemComplexity
from pyccel.complexity.arithmetic import OpComplexity
import os
import pytest


base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]


@pytest.mark.parametrize( "f", files )
def test_complexity(f):

    mem_complexity = MemComplexity(f)
    arithmetic_complexity = OpComplexity(f)
    print(mem_complexity.cost())
    print(arithmetic_complexity.cost())

######################
if __name__ == '__main__':

    print('*********************************')
    print('***                           ***')
    print('***     TESTING COMPLEXITY    ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_complexity(f)
        print("\n")
