import ast
from _ast import *
with open('pyccel/ast/basic.py','r') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, (FunctionDef, ClassDef, Module)):
        if isinstance(node, Module):
            pass
        if isinstance(node, ast.FunctionDef):
            if hasattr(node, "parent"):
                print(node.parent.name, node.name, sep='.')
            else:
                print(node.name)
        if isinstance(node, ast.ClassDef):
            print(node.name)
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    child.parent = node