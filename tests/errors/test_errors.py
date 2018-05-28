# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO - syntax errors tests
#      - expected errors in log files for every script

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.parser.errors import Errors
import os

def test_syntax_errors():
    print('*********************************')
    print('***                           ***')
    print('***   TESTING SYNTAX ERRORS   ***')
    print('***                           ***')
    print('*********************************')

    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'syntax')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))

        pyccel = Parser(f)

        try:
            ast = pyccel.parse()
        except:
            pass

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    os.chdir(init_dir)
    print('\n')

def test_semantic_errors():
    print('*********************************')
    print('***                           ***')
    print('***  TESTING SEMANTIC ERRORS  ***')
    print('***                           ***')
    print('*********************************')

    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'semantic')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
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

    os.chdir(init_dir)
    print('\n')

######################
if __name__ == '__main__':
    test_syntax_errors()
    test_semantic_errors()
