# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO - syntax errors tests
#      - expected errors in log files for every script

import os
import pytest

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.codegen.pipeline import execute_pyccel
from pyccel.errors.errors   import Errors, PyccelSyntaxError, PyccelSemanticError, PyccelCodegenError, PyccelError
from pyccel.errors.errors   import ErrorsMode
from pyccel.naming                 import name_clash_checkers
from pyccel.parser.scope           import Scope

error_mode = ErrorsMode()


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

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    pyccel = Parser(f, output_folder = os.getcwd())

    with pytest.raises(PyccelSyntaxError):
        pyccel.parse(verbose = 0)

    assert errors.has_blockers()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.parametrize("f",get_files_from_folder("syntax_errors"))
def test_syntax_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    pyccel = Parser(f, output_folder = os.getcwd())

    pyccel.parse(verbose = 0)

    assert errors.has_errors()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.parametrize("f", get_files_from_folder("semantic/blocking"))
def test_semantic_blocking_errors(f):
    print(f'> testing {f}')

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    Scope.name_clash_checker = name_clash_checkers['fortran']
    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    with pytest.raises(PyccelSemanticError):
        pyccel.annotate(verbose = 0)

    assert errors.has_blockers()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.xdist_incompatible
def test_traceback():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    f = os.path.join(base_dir, 'semantic/blocking/INHOMOG_LIST.py')
    print(f'> testing {f}')

    # reset Errors singleton
    errors = Errors()
    errors.reset()
    error_mode.set_mode('developer')

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    try:
        pyccel.annotate(verbose = 0)
    except PyccelSemanticError as e:
        msg = str(e)
        errors.report(msg,
            severity='error',
            traceback=e.__traceback__)

    assert errors.has_blockers()
    assert errors.num_messages() == 2
    error_mode.set_mode('user')

semantic_non_blocking_errors_args = [f for f in get_files_from_folder("semantic/non_blocking")]
@pytest.mark.parametrize("f", semantic_non_blocking_errors_args)
def test_semantic_non_blocking_errors(f):
    print(f'> testing {f}')

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    pyccel.annotate(verbose = 0)

    assert errors.has_errors()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.xdist_incompatible
@pytest.mark.parametrize("f", semantic_non_blocking_errors_args)
def test_semantic_non_blocking_developer_errors(f):
    print(f'> testing {f}')

    # reset Errors singleton
    errors = Errors()
    errors.reset()
    error_mode.set_mode('developer')

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    with pytest.raises(PyccelSemanticError):
        pyccel.annotate(verbose = 0)

    error_mode.set_mode('user')
    assert errors.has_errors()

@pytest.mark.parametrize("f",get_files_from_folder("codegen/fortran_blocking"))
def test_codegen_blocking_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    ast = pyccel.annotate(verbose = 0)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, 'fortran', verbose = 0)
    with pytest.raises(PyccelCodegenError):
        codegen.printer.doprint(codegen.ast)

    assert errors.has_errors()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.parametrize("f",get_files_from_folder("codegen/fortran_non_blocking"))
def test_codegen_non_blocking_errors(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    ast = pyccel.annotate(verbose = 0)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, 'fortran', verbose = 0)
    codegen.printer.doprint(codegen.ast)

    assert errors.has_errors()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

@pytest.mark.parametrize("f",get_files_from_folder("known_bugs"))
def test_neat_errors_for_known_bugs(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()

    with open(f, encoding='utf-8') as fl:
        expected_error_msg = fl.readlines()[0][1:].strip()

    with pytest.raises(PyccelError):
        execute_pyccel(f)

    assert errors.has_errors()
    messages = [str(e.message) for f_errs in errors.error_info_map.values() for e in f_errs]
    assert any(expected_error_msg in m for m in messages)

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
