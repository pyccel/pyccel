# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os
import pytest
import astunparse

from pyccel.parser.syntactic import SyntaxParser
from pyccel.errors.errors    import Errors

class Unparser(astunparse.Unparser):
    """Unparser the AST"""

    def _CommentLine(self, node):
        self.write('\n' + ' '*node.col_offset + node.s)

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) \
         #if f not in failing_files \
         #else pytest.param(os.path.join(path_dir,f), marks = pytest.mark.xfail(reason=failing_files[f])) \
         for f in files \
         if f.endswith(".py") \
        ]
@pytest.mark.fortran
@pytest.mark.parametrize("f", files)
def test_parse(f):

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    with open(f) as infile:
        orig = infile.read().strip()

    pyccel = SyntaxParser(f)
    with open('out.py','w') as outfile:
        u = Unparser(pyccel.fst, file=outfile)

    with open('out.py') as infile:
        copy = infile.read().strip()

    orig_lines = orig.expandtabs(4).split('\n')
    copy_lines = copy.expandtabs(4).split('\n')

    for o,c in zip(orig_lines,copy_lines):
        # Check for the same indentation
        for oi, ci in zip(o,c):
            if oi == ' ' or ci == ' ':
                assert(oi==ci)
            else:
                break
        o = o.replace(' ','')
        c = c.replace(' ','')
        assert(o==c)

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***   TESTING CODEGEN FCODE   ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_parse(f)

    print('\n')

