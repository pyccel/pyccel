# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.complexity.memory import MemComplexity
import os

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

# ==============================================================================
def test_complexity(f, mode=None):

    complexity = MemComplexity(f)
    print(complexity.cost(mode=mode))
    print('----------------------')
    for f, c in complexity.costs.items():
        print('> cost of {} = {}'.format(f, c))

######################
if __name__ == '__main__':

#    test_complexity('tmp.py')

    print('*********************************')
    print('***                           ***')
    print('***     TESTING COMPLEXITY    ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_complexity(f)
        print("\n")
