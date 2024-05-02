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
    
@pytest.mark.parametrize( "f", files )
def test_codegen(f):

    print('> testing {0}'.format(str(f)))

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    code = codegen.doprint()

    # reset Errors singleton
    errors = Errors()
    errors.reset()

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***   TESTING CODEGEN FCODE   ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        test_codegen(f)

    print('\n')
