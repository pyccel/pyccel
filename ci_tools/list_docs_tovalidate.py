""" Script to list the objects that should be numpydoc validated
"""
import ast
from _ast import FunctionDef, ClassDef, Module
import argparse

parser = argparse.ArgumentParser(description='List the objects with docstrings in the files provided')
parser.add_argument('files', metavar='files', type=str,
                        help='File containing the list of files modified by the PR')
parser.add_argument('output', metavar='output', type=str,
                        help='File to save the output to')
args = parser.parse_args()

with open(args.files,'r', encoding="utf-8") as f:
    lines = f.readlines()

lines = [l.strip() for l in lines[:-1]]
changes = {}
for l in lines:
    file, line_no = l.split()
    if file in changes:
        changes[file].append(int(line_no))
    else:
        changes[file] = []
        changes[file].append(int(line_no))

objects = []
for file in changes:
    with open(file,'r', encoding="utf-8") as f:
        tree = ast.parse(f.read())
    prefix = file[:-3].replace('/', '.')
    for node in ast.walk(tree):
        if isinstance(node, (FunctionDef, ClassDef, Module)):
            if isinstance(node, Module):
                objects.append(prefix)
            if isinstance(node, FunctionDef):
                if hasattr(node, "parent"):
                    if any([x >= node.lineno and x <= node.end_lineno
                            for x in changes[file]]):
                        objects.append('.'.join([prefix,node.parent.name, node.name]))
                else:
                    if any([x >= node.lineno and x <= node.end_lineno
                            for x in changes[file]]):
                        objects.append('.'.join([prefix, node.name]))
            if isinstance(node, ClassDef):
                if any([x >= node.lineno and x <= node.end_lineno
                        for x in changes[file]]):
                    objects.append('.'.join([prefix, node.name]))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        child.parent = node
    with open(args.output, 'a', encoding="utf-8") as f:
        for obj in objects:
            print(obj, file=f)
