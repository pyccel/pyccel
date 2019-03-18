from pyccel.complexity.memory import MemComplexity
from pyccel.complexity.arithmetic import OpComplexity
import os
def test_complexity():
    print('*********************************')
    print('***                           ***')
    print('***     TESTING COMPLEXITY    ***')
    print('***                           ***')
    print('*********************************')

    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))
        mem_complexity = MemComplexity(f)
        arithmetic_complexity = OpComplexity(f)
        print(mem_complexity.cost())
        print(arithmetic_complexity.cost())











######################
if __name__ == '__main__':
    test_complexity()
