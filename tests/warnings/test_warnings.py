# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

# TODO warnings for syntax and semantic stages

from pyccel.parser import Parser
from pyccel.codegen import Codegen
from pyccel.parser.errors import Errors
import os

#def test_syntax_warnings():
#    print('*********************************')
#    print('***                           ***')
#    print('***  testing syntax warnings  ***')
#    print('***                           ***')
#    print('*********************************')
#
#    init_dir = os.getcwd()
#    base_dir = os.path.dirname(os.path.realpath(__file__))
#    path_dir = os.path.join(base_dir, 'syntax')
#
#    files = sorted(os.listdir(path_dir))
#    files = [f for f in files if (f.endswith(".py"))]
#
#    os.chdir(path_dir)
#    for f in files:
#        print('> testing {0}'.format(str(f)))
#
#        pyccel = Parser(f)
#
#        try:
#            ast = pyccel.parse()
#        except:
#            pass
#
#        # reset Errors singleton
#        errors = Errors()
#        errors.reset()
#
#    os.chdir(init_dir)
#    print('\n')
#
#def test_semantic_warnings():
#    print('*********************************')
#    print('***                           ***')
#    print('*** testing semantic warnings ***')
#    print('***                           ***')
#    print('*********************************')
#
#    init_dir = os.getcwd()
#    base_dir = os.path.dirname(os.path.realpath(__file__))
#    path_dir = os.path.join(base_dir, 'semantic')
#
#    files = sorted(os.listdir(path_dir))
#    files = [f for f in files if (f.endswith(".py"))]
#
#    os.chdir(path_dir)
#    for f in files:
#        print('> testing {0}'.format(str(f)))
#
#        pyccel = Parser(f, show_traceback=False)
#        ast = pyccel.parse()
#
#        try:
#            settings = {}
#            ast = pyccel.annotate(**settings)
#        except:
#            pass
#
#        # reset Errors singleton
#        errors = Errors()
#        errors.reset()
#
#    os.chdir(init_dir)
#    print('\n')

def test_codegen_warnings():
    print('*********************************')
    print('***                           ***')
    print('***  testing codegen warnings ***')
    print('***                           ***')
    print('*********************************')

    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'codegen')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))

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

    os.chdir(init_dir)
    print('\n')

######################
if __name__ == '__main__':
#    test_syntax_warnings()
#    test_semantic_warnings()
    test_codegen_warnings()
