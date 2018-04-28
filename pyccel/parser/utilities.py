# coding: utf-8

"""This file contains different utilities for the Parser."""

from redbaron import (CommentNode, ForNode, DefNode, WithNode,
                      IfNode, ElseNode, ElifNode, IfelseblockNode,
                      EndlNode)

from sympy import srepr, sympify
from sympy.printing.dot import dotprint
import os


def read_file(filename):
    """Returns the source code from a filename."""
    f = open(filename)
    code = f.read()
    f.close()
    return code

#  ... checking the validity of the filenames, using absolute paths
def _is_valid_filename(filename, ext):
    """Returns True if filename has the extension ext and exists."""
    if not isinstance(filename, str):
        return False

    if not(ext == filename.split('.')[-1]):
        return False
    fname = os.path.abspath(filename)
    return os.path.isfile(fname)

def is_valid_filename_py(filename):
    """Returns True if filename is an existing python file."""
    return _is_valid_filename(filename, 'py')

def is_valid_filename_pyh(filename):
    """Returns True if filename is an existing pyccel header file."""
    return _is_valid_filename(filename, 'pyh')
#  ...

#  ...
def header_statement(stmt, accel):
    """Returns stmt if a header statement. otherwise it returns None.
    this function can be used as the following
    >>> if header_statement(stmt):
        # do stuff
        ...

    """
    if not isinstance(stmt, CommentNode): None
    if not stmt.value.startswith('#$'): None

    header = stmt.value[2:].lstrip()
    if not directive.startswith('header'): None

    return stmt.value
#  ...

# ... utilities for parsing OpenMP/OpenACC directives
def accelerator_statement(stmt, accel):
    """Returns stmt if an accelerator statement. otherwise it returns None.
    this function can be used as the following
    >>> if accelerator_statement(stmt, 'omp'):
        # do stuff
        ...

    In general you can use the functions omp_statement and acc_statement
    """
    assert(accel in ['omp', 'acc'])

    if not isinstance(stmt, CommentNode): None
    if not stmt.value.startswith('#$'): None

    directive = stmt.value[2:].lstrip()
    if not directive.startswith(accel): None

    return stmt.value

omp_statement = lambda x: accelerator_statement(x, 'omp')
acc_statement = lambda x: accelerator_statement(x, 'acc')
# ...

# ... preprocess fst for comments
get_comments = lambda y: y.filter(lambda x: isinstance(x, CommentNode))
get_loops = lambda y: y.filter(lambda x: isinstance(x, ForNode))
get_defs = lambda y: y.filter(lambda x: isinstance(x, DefNode))
get_withs = lambda y: y.filter(lambda x: isinstance(x, WithNode))
get_ifs = lambda y: y.filter(lambda x: isinstance(x, (IfNode, ElseNode, ElifNode)))
get_ifblocks = lambda y: y.filter(lambda x: isinstance(x, IfelseblockNode))

def fst_move_directives(x):
    """This function moves OpenMP/OpenAcc directives from loop statements to
    their appropriate parent. This function will have inplace effect.
    In order to understand why it is needed, let's take a look at the following
    exampe

    >>> code = '''
    ... #$ omp do schedule(runtime)
    ... for i in range(0, n):
    ...     for j in range(0, m):
    ...         a[i,j] = i-j
    ... #$ omp end do nowait
    ... '''
    >>> from redbaron import RedBaron
    >>> red = RedBaron(code)
    >>> red
    0   '\n'
    1   #$ omp do schedule(runtime)
    2   for i in range(0, n):
            for j in range(0, m):
                a[i,j] = i-j
        #$ omp end do nowait

    As you can see, the statement `#$ omp end do nowait` is inside the For
    statement, while we would like to have it outside.
    Now, let's apply our function

    >>> fst_move_directives(red)
    0   #$ omp do schedule(runtime)
    1   for i in range(0, n):
            for j in range(0, m):
                a[i,j] = i-j
    2   #$ omp end do nowait
    """
    # ... def and with statements
    defs = get_defs(x)
    withs = get_withs(x)
    containers = defs + withs
    for stmt in containers:
        fst_move_directives(stmt.value)
    # ...

    # ... if statements are inside IfelseblockNode
    ifblocks = get_ifblocks(x)
    for ifblock in ifblocks:
        for stmt in ifblock.value:
            fst_move_directives(stmt.value)
    # ...

    # ... loops
    xs = get_loops(x)
    for son in xs:
        fst_move_directives(son)

        cmts = get_comments(son)
        # we only take comments that are using OpenMP/OpenAcc
        cmts = [i for i in cmts if omp_statement(i) or acc_statement(i)]
        for cmt in cmts:
            son.value.remove(cmt)

        # insert right after the loop
        i_son = x.index(son)
        for i,cmt in enumerate(cmts):
            son.parent.insert(i_son+i+1, cmt)
    # ...

    return x
# ...

# ...
def reconstruct_pragma_multilines(header):
    """Must be called once we visit an annotated comment, to get the remaining
    parts of a statement written on multiple lines."""

    def _is_pragma(x):
        if not(isinstance(x, CommentNode) and x.value.startswith('#$')):
            return False
        env = x.value[2:].lstrip()
        if (env.startswith('header') or
            env.startswith('omp') or
            env.startswith('acc')):
            return False
        return True

    _ignore_stmt = lambda x: isinstance(x, (EndlNode, CommentNode)) and not _is_pragma(x)

    ls = []
    node = header.next
    while _is_pragma(node) or _ignore_stmt(node):
        # append the pragma stmt
        if _is_pragma(node):
            ls.append(node.value)

        # look if there are comments or empty lines
        node = node.next
        if _ignore_stmt(node):
            node = node.next

    txt = ' '.join(i for i in ls)
    txt = txt.replace('#$', '')
    txt = '{} {}'.format(header.value, txt)
    return txt
# ...


#  ... utilities
def view_tree(expr):
    """Views a sympy expression tree."""
    print (srepr(expr))
#  ...

