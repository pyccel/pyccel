# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO - syntax errors tests
#      - expected errors in log files for every script

import os
import pytest

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors, PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError, PyccelError


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

    assert(errors.has_blockers())

@pytest.mark.parametrize("f",get_files_from_folder("syntax_errors"))
def test_syntax_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)

    ast = pyccel.parse()

    assert(errors.has_errors())

@pytest.mark.parametrize("f",get_files_from_folder("semantic/blocking"))
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

    assert(errors.has_blockers())

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

    assert(errors.has_errors())


@pytest.mark.parametrize("f",get_files_from_folder("codegen/fortran"))
def test_codegen_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    with pytest.raises(PyccelCodegenError):
        codegen.doprint()

    assert(errors.has_errors())

@pytest.mark.parametrize("f",get_files_from_folder("known_bugs"))
def test_neat_errors_for_known_bugs(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)
    with pytest.raises(PyccelError):
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        name = os.path.basename(f)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name)
        codegen.doprint()

    assert(errors.has_errors())

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***   TESTING SYNTAX ERRORS   ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder("syntax_errors"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_syntax_errors(f)
    for f in get_files_from_folder("syntax_blockers"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_syntax_blockers(f)
    print('\n')

    print('*********************************')
    print('***                           ***')
    print('***  TESTING SEMANTIC ERRORS  ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder("semantic/non_blocking"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_semantic_non_blocking_errors(f)
    for f in get_files_from_folder("semantic/blocking"):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_semantic_blocking_errors(f)
    print('\n')
