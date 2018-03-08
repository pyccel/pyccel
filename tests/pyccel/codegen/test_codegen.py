# coding: utf-8

from pyccel.parser import Parser
import os

from pyccel.codegen import Codegen

def test_codegen():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    #path_dir = os.path.join(base_dir, '../parser/scripts/semantic')
    path_dir = os.path.join(base_dir, 'scripts')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    for f in files:
        print('> testing {0}'.format(str(f)))
        f_name = os.path.join(path_dir, f)

        pyccel = Parser(f_name)
        ast = pyccel.parse()

        settings = {}
        ast = pyccel.annotate(**settings)

        name = os.path.basename(f_name)
        name = os.path.splitext(name)[0]

        codegen = Codegen(ast, name)
        code = codegen.doprint()
#        print(code)

######################
if __name__ == '__main__':
    test_codegen()
