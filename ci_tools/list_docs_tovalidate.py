""" Script to list the objects that should be numpydoc validated
"""
import ast
from _ast import FunctionDef, ClassDef, Module
import argparse
from git_evaluation_tools import get_diff_as_json as gdj

parser = argparse.ArgumentParser(description='List the objects with docstrings in the files provided')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                        help='Output of git diff between PR branch and base branch')
parser.add_argument('output', metavar='output', type=str,
                        help='File to save the output to')
args = parser.parse_args()

results = gdj(args.gitdiff)
changes = {}
for file,changes in results.items():
    if file.startswith('pyccel/'):
        for line_no in changes['addition']:
            if file in changes:
                changes[file].append(int(line_no))
            else:
                changes[file] = []
                changes[file].append(int(line_no))

objects = []
for file, line_nos in changes.items():
    with open(file,'r', encoding="utf-8") as f:
        tree = ast.parse(f.read())
    prefix = file[:-3].replace('/', '.')
    for node in ast.walk(tree):
        if isinstance(node, (FunctionDef, ClassDef, Module)):
            if isinstance(node, Module):
                objects.append(prefix)
            if isinstance(node, FunctionDef):
                if hasattr(node, "parent"):
                    if any((node.lineno <= x <= node.end_lineno
                            for x in line_nos)):
                        objects.append('.'.join([prefix,node.parent.name, node.name]))
                else:
                    if any((node.lineno <= x <= node.end_lineno
                            for x in line_nos)):
                        objects.append('.'.join([prefix, node.name]))
            if isinstance(node, ClassDef):
                if any((node.lineno <= x <= node.end_lineno
                        for x in line_nos)):
                    objects.append('.'.join([prefix, node.name]))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        child.parent = node
    with open(args.output, 'a', encoding="utf-8") as f:
        for obj in objects:
            print(obj, file=f)
