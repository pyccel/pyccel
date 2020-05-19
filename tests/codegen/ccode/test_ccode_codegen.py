# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.parser.errors import Errors
import pytest
import os

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]


failing_files = [os.path.join(path_dir,'arrays.py')]
passing_files = list(set(files).difference(set(failing_files)))
    
def codegen_test(f):

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    code = codegen.doprint(language='c')

    # reset Errors singleton
    errors = Errors()
    errors.reset()

@pytest.mark.c
@pytest.mark.parametrize( "f", passing_files )
def test_passing_codegen(f):
    codegen_test(f)

@pytest.mark.c
@pytest.mark.xfail
@pytest.mark.parametrize( "f", failing_files )
def test_failing_codegen(f):
    codegen_test(f)

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING CODEGEN CCODE   ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        codegen_test(f)

    print('\n')

