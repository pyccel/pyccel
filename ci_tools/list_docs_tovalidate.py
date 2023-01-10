""" Script to list the objects that should be numpydoc validated
"""
import ast
from _ast import *
import argparse

parser = argparse.ArgumentParser(description='List the objects with docstrings in the files provided')
parser.add_argument('files', metavar='files', type=str,
                        help='File containing the list of files modified by the PR')
parser.add_argument('output', metavar='output', type=str,
                        help='File to save the output to')
args = parser.parse_args()

with open(args.files,'r') as f:
    files = f.readlines()

files = [l.strip() for l in files[:-1]]
objects = []
for file in files:
    with open(file,'r') as f:
        tree = ast.parse(f.read())
    prefix = file[:-3].replace('/', '.')
    for node in ast.walk(tree):
        if isinstance(node, (FunctionDef, ClassDef, Module)):
            if isinstance(node, Module):
                objects.append(prefix)
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, "parent"):
                    objects.append('.'.join([prefix,node.parent.name, node.name]))
                else:
                    objects.append('.'.join([prefix, node.name]))
            if isinstance(node, ast.ClassDef):
                objects.append('.'.join([prefix, node.name]))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        child.parent = node
    with open(args.output, 'a', encoding="utf-8") as f:
        for obj in objects:
            print(obj, file=f)
