# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO - syntax errors tests
#      - expected errors in log files for every script

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.parser.errors import Errors, PyccelSemanticError, PyccelSyntaxError
import os
import pytest

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join(foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

@pytest.mark.parametrize("f",get_files_from_folder("syntax_blockers"))
def test_syntax_blockers(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)

    with pytest.raises(PyccelSyntaxError):
        ast = pyccel.parse()

    assert(errors.is_blockers())

@pytest.mark.parametrize("f",get_files_from_folder("syntax_errors"))
def test_syntax_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)

    ast = pyccel.parse()

    assert(errors.is_errors())

semantic_blocking_xfails = {'ex6.py':'different shape not recognised as different type : issue 325'}
semantic_blocking_errors_args = [f if os.path.basename(f) not in semantic_blocking_xfails \
                          else pytest.param(f, marks = pytest.mark.xfail(reason=semantic_blocking_xfails[os.path.basename(f)])) \
                          for f in get_files_from_folder("semantic/blocking")]
@pytest.mark.parametrize("f", semantic_blocking_errors_args)
def test_semantic_blocking_errors(f):
    print('> testing {0}'.format(str(f)))

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f, show_traceback=False)
    ast = pyccel.parse()

    settings = {}
    with pytest.raises(PyccelSemanticError):
        ast = pyccel.annotate(**settings)

    assert(errors.is_errors())

semantic_non_blocking_errors_args = [f for f in get_files_from_folder("semantic/non_blocking")]
@pytest.mark.parametrize("f", semantic_non_blocking_errors_args)
def test_semantic_non_blocking_errors(f):
    print('> testing {0}'.format(str(f)))

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f, show_traceback=False)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    assert(errors.is_errors())


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
