""" Script to list the objects that should be numpydoc validated
"""
import ast
from _ast import FunctionDef, ClassDef
import argparse
from pathlib import PurePath
from git_evaluation_tools import get_diff_as_json

parser = argparse.ArgumentParser(description='List the objects with docstrings in the files provided')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                        help='Output of git diff between PR branch and base branch')
parser.add_argument('output', metavar='output', type=str,
                        help='File to save the output to')
args = parser.parse_args()

results = get_diff_as_json(args.gitdiff)
changes = {}
for file, upds in results.items():
    if PurePath(file).parts[0] == 'pyccel':
        for line_no in upds['addition']:
            if file in changes:
                changes[file].append(int(line_no))
            else:
                changes[file] = [int(line_no)]

objects = []
for file, line_nos in changes.items():
    with open(file,'r', encoding="utf-8") as f:
        tree = ast.parse(f.read())
    # The prefix variable stores the absolute dotted prefix of all
    # objects in the file.
    # for example: the objects in the file pyccel/ast/core.py will
    # have the prefix pyccel.ast.core
    prefix = '.'.join(PurePath(file).with_suffix('').parts)

    for node in ast.walk(tree):
        # This loop walks the ast and explores all objects
        # present in the file.
        # If the object is an instance of a FunctionDef or
        # a ClassDef, a check is performed to see if any of
        # the updated lines are present within the object.
        # Additionally, The name of all objects present
        # inside a function or a class is updated to include
        # the name of the parent object
        if isinstance(node, (FunctionDef, ClassDef)):
            if any((node.lineno <= x <= node.end_lineno
                    for x in line_nos)) and not node.name.startswith('inner_function'):
                objects.append('.'.join([prefix, node.name]))
            obj_pref = node.name if isinstance(node, ClassDef) else 'inner_function'
            for child in node.body:
                if isinstance(child, (FunctionDef, ClassDef)):
                    child.name = '.'.join([obj_pref, child.name])

    with open(args.output, 'a', encoding="utf-8") as f:
        for obj in objects:
            print(obj, file=f)
