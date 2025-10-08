# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

from ast import _Unparser

import os
import pytest

from pyccel.parser.syntactic import SyntaxParser
from pyccel.errors.errors    import Errors

class Unparser(_Unparser):
    """Unparser the AST"""

    def visit_CommentLine(self, node):
        self.fill(node.s)

    def visit_CommentMultiLine(self, node):
        self.fill(node.s)

def get_indent(l):
    i = 0
    for li in l:
        if li == ' ':
            i+=1
        else:
            break
    return i

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

    pyccel = SyntaxParser(f, verbose=0)
    unparser = Unparser()
    copy = unparser.visit(pyccel.fst)

    # Get non-empty lines
    orig_lines = [l for l in orig.expandtabs(4).split('\n') if l!= '']
    copy_lines = [l for l in copy.expandtabs(4).split('\n') if l!= '']

    # Get indentation
    orig_indents = [get_indent(l) for l in orig_lines]
    copy_indents = [get_indent(l) for l in copy_lines]

    on = len(orig_lines)
    cn = len(copy_lines)
    oi = 0
    ci = 0
    while oi<on and ci<cn:
        o = orig_lines[oi]
        c = copy_lines[ci]
        # Check for the same indentation
        assert orig_indents[oi] == copy_indents[ci]
        # Handle change between elif and else
        if o.startswith('else') and c.startswith('elif'):
            oi += 1
            c = c[2:]
            # Change indentation to match
            indent = orig_indents[oi]
            diff = (indent-orig_indents[oi-1])
            i = oi
            while i<len(orig_indents) and orig_indents[i]>=indent:
                orig_indents[i]-=diff
                i+=1
        elif o.startswith('elif') and c.startswith('else'):
            ci += 1
            o = o[2:]
            # Change indentation to match
            indent = copy_indents[ci]
            diff = (indent-copy_indents[ci-1])
            i = ci
            while i<len(copy_indents) and copy_indents[i]>=indent:
                copy_indents[i]-=diff
                i+=1
        # Check lines
        assert o==c
        oi += 1
        ci += 1

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

