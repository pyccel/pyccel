# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO warnings for syntax and semantic stages

import os
import pytest

from wrapper import HIGH_ORDER_FUNCTIONS_IN_CLASS_FUNCS

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors
from pyccel                 import epyccel

def get_files_from_folder(foldername):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, os.path.join(foldername))

    files = sorted(os.listdir(path_dir))
    files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]
    return files

#@pytest.mark.language_agnostic
#@pytest.mark.parametrize("f",get_files_from_folder('syntax'))
#def test_syntax_warnings(f):
#
#    # reset Errors singleton
#    errors = Errors()
#    errors.reset()
#
#    pyccel = Parser(f, output_folder = os.getcwd())
#
#    ast = pyccel.parse(verbose = 0)
#
#    assert errors.num_messages()!=0

@pytest.mark.language_agnostic
@pytest.mark.xdist_incompatible
@pytest.mark.parametrize("f",get_files_from_folder('semantic'))
def test_semantic_warnings(f):

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    pyccel.annotate(verbose = 0)

    assert not errors.has_errors()
    assert errors.has_warnings()

#@pytest.mark.parametrize("f", codegen_errors_args)
#def test_codegen_warnings(f):
#
#    # reset Errors singleton
#    errors = Errors()
#    errors.reset()
#
#    pyccel = Parser(f, output_folder = os.getcwd())
#    ast = pyccel.parse(verbose = 0)
#
#    ast = pyccel.annotate(verbose = 0)
#
#    name = os.path.basename(f)
#    name = os.path.splitext(name)[0]
#
#    codegen = Codegen(ast, name, 'fortran', verbose = 0)
#    code = codegen.printer.doprint(codegen.ast)
#
#    assert errors.has_warnings()
#    assert not errors.has_errors()


@pytest.mark.parametrize("f", [HIGH_ORDER_FUNCTIONS_IN_CLASS_FUNCS])
@pytest.mark.c
def test_cwrapper_warnings(f):
    with pytest.warns(UserWarning):
        epyccel(f, language='c')

@pytest.mark.parametrize("f", [HIGH_ORDER_FUNCTIONS_IN_CLASS_FUNCS])
@pytest.mark.fortran
def test_bind_c_warnings(f):
    with pytest.warns(UserWarning):
        epyccel(f, language='fortran')

######################
if __name__ == '__main__':
#    print('*********************************')
#    print('***                           ***')
#    print('***  testing syntax warnings  ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('syntax'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_syntax_warnings(f)
#    print('\n')
    print('*********************************')
    print('***                           ***')
    print('*** testing semantic warnings ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('semantic'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_semantic_warnings(f)
    print('\n')
#    print('*********************************')
#    print('***                           ***')
#    print('***  testing codegen warnings ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('codegen'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_codegen_warnings(f)
#    print('\n')
