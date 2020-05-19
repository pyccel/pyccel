from pyccel.complexity.memory import MemComplexity
from pyccel.complexity.arithmetic import OpComplexity
import os
import pytest


base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [f for f in files if (f.endswith(".py"))]


@pytest.mark.parametrize( "f", files )
def test_complexity(f):

    os.chdir(path_dir)

    print('> testing {0}'.format(str(f)))
    mem_complexity = MemComplexity(f)
    arithmetic_complexity = OpComplexity(f)
    print(mem_complexity.cost())
    print(arithmetic_complexity.cost())

    os.chdir(os.getcwd())

######################
if __name__ == '__main__':

    print('*********************************')
    print('***                           ***')
    print('***     TESTING COMPLEXITY    ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        test_complexity(f)
        print("\n")
