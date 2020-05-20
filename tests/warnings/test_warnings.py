# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO warnings for syntax and semantic stages

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

#@pytest.mark.parametrize("f",get_files_from_folder('syntax'))
#def test_syntax_warnings(f):
#
#    pyccel = Parser(f)
#
#    try:
#        ast = pyccel.parse()
#    except:
#        pass
#
#    # reset Errors singleton
#    errors = Errors()
#    errors.reset()
#
#@pytest.mark.parametrize("f",get_files_from_folder('semantic'))
#def test_semantic_warnings():
#
#    pyccel = Parser(f, show_traceback=False)
#    ast = pyccel.parse()
#
#    try:
#        settings = {}
#        ast = pyccel.annotate(**settings)
#    except:
#        pass
#
#    # reset Errors singleton
#    errors = Errors()
#    errors.reset()

@pytest.mark.parametrize("f",get_files_from_folder('codegen'))
def test_codegen_warnings(f):

    pyccel = Parser(f, show_traceback=False)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    try:
        name = os.path.basename(f)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name)
        code = codegen.doprint()
    except:
        pass

    # reset Errors singleton
    errors = Errors()
    errors.reset()

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
#    print('*********************************')
#    print('***                           ***')
#    print('*** testing semantic warnings ***')
#    print('***                           ***')
#    print('*********************************')
#    for f in get_files_from_folder('semantic'):
#        print('> testing {0}'.format(str(os.path.basename(f))))
#        test_semantic_warnings(f)
#    print('\n')
    print('*********************************')
    print('***                           ***')
    print('***  testing codegen warnings ***')
    print('***                           ***')
    print('*********************************')
    for f in get_files_from_folder('codegen'):
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_codegen_warnings(f)
    print('\n')
