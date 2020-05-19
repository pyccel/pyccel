# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO - syntax errors tests
#      - expected errors in log files for every script

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.parser.errors import Errors
import os
import pytest

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join(foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

@pytest.mark.parametrize("f",get_files_from_folder("syntax"))
def test_syntax_errors(f):
    pyccel = Parser(f)

    try:
        ast = pyccel.parse()
    except:
        pass

    # reset Errors singleton
    errors = Errors()
    errors.reset()

@pytest.mark.parametrize("f",get_files_from_folder("semantic"))
def test_semantic_errors(f):
    print('> testing {0}'.format(str(f)))

    pyccel = Parser(f, show_traceback=False)
    ast = pyccel.parse()

    try:
        settings = {}
        ast = pyccel.annotate(**settings)
    except:
        pass

    # reset Errors singleton
    errors = Errors()
    errors.reset()

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***   TESTING SYNTAX ERRORS   ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder("syntax"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_syntax_errors(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING SEMANTIC ERRORS  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder("semantic"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_semantic_errors(f)
    print('\n')
