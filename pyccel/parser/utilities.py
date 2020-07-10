# coding: utf-8

"""This file contains different utilities for the Parser."""

from redbaron import (ForNode, DefNode, WithNode,
                      IfNode, ElseNode, ElifNode, IfelseblockNode,
                      EndlNode, DottedAsNameNode, NameNode,
                      CallNode, RedBaron, AtomtrailersNode)

from pyccel.parser.extend_tree import CommentLine

from sympy import srepr
from pyccel.ast.core import DottedName
from pyccel.ast.core import create_variable
from sympy import Symbol
import os

import string
import random

pyccel_external_lib = {"mpi4py"             : "pyccel.stdlib.external.mpi4py",
                       "scipy.linalg.lapack": "pyccel.stdlib.external.lapack",
                       "scipy.linalg.blas"  : "pyccel.stdlib.external.blas",
                       "scipy.fftpack"      : "pyccel.stdlib.external.dfftpack",
                       "fitpack"            : "pyccel.stdlib.internal.fitpack",
                       "scipy.interpolate._fitpack":"pyccel.stdlib.external.fitpack"}

#==============================================================================
def random_string( n ):
    # we remove uppercase letters because of f2py
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================

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
    if not isinstance(stmt, CommentLine): return None
    if not stmt.value.startswith('#$'): return None

    header = stmt.value[2:].lstrip()
    if not directive.startswith('header'): return None

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

    if not isinstance(stmt, CommentLine): return None
    if not stmt.value.startswith('#$'): return None

    directive = stmt.value[2:].lstrip()
    if not directive.startswith(accel): return None

    return stmt.value

omp_statement = lambda x: accelerator_statement(x, 'omp')
acc_statement = lambda x: accelerator_statement(x, 'acc')
# ...

def get_module_name( dotted_as_node ):
    code_name = dotted_as_node.target
    if (code_name != ""):
        return [ code_name ]
    else:
        import_name = dotted_as_node.value
        return import_name.dumps().split('.')

# ...
def reconstruct_pragma_multilines(header):
    """Must be called once we visit an annotated comment, to get the remaining
    parts of a statement written on multiple lines."""

    # ...
    def _is_pragma(x):
        if not(isinstance(x, CommentLine) and x.value.startswith('#$')):
            return False
        env = x.value[2:].lstrip()
        if (env.startswith('header') or
            env.startswith('omp') or
            env.startswith('acc')):
            return False
        return True

    _ignore_stmt = lambda x: isinstance(x, (EndlNode, CommentLine)) and not _is_pragma(x)
    def _is_multiline(x):
        # we use tr/except to avoid treating nodes without .value
        try:
            return x.s.rstrip().endswith('&')
        except AttributeError:
            return False

    condition = lambda x: (_is_multiline(x.parent) and (_is_pragma(x) or _ignore_stmt(x)))
    # ...

    if not _is_multiline(header):
        return header.s

    ls = []
    node = header.next
    while condition(node):
        # append the pragma stmt
        if _is_pragma(node):
            ls.append(node.value)

        # look if there are comments or empty lines
        node = node.next
        if _ignore_stmt(node):
            node = node.next

    txt = ' '.join(i for i in ls)
    txt = txt.replace('#$', '')
    txt = txt.replace('&', '')
    txt = '{} {}'.format(header.value.replace('&', ''), txt)
    return txt
# ...


#  ... utilities
def view_tree(expr):
    """Views a sympy expression tree."""
    print (srepr(expr))
#  ...

def get_default_path(name):
   """this function takes a an import name
      and returns the path full bash of the library
      if the library is in stdlib"""
   name_ = name
   if isinstance(name, (DottedName, Symbol)):
       name_ = str(name)
   if name_ in pyccel_external_lib.keys():
        name = pyccel_external_lib[name_].split('.')
        if len(name)>1:
            return DottedName(*name)
        else:
            return name[0]
   return name


