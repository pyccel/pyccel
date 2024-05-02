""" Script to check that Pyccel coding conventions are correctly followed in the AST
"""
import argparse
import importlib
import inspect
import os
import sys
from pyccel import ast
from pyccel.ast.basic import Basic, PyccelAstNode, ScopedNode

parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

# Get ast modules
ast_folder = os.path.dirname(ast.__file__)
ast_modules = [mod[:-3] for mod in os.listdir(ast_folder) if mod != '__init__.py' and mod.endswith('.py')]

# Prepare error collection
missing_all = []
non_alphabetical_all = []
missing_slots = []
overridden_slots = []
missing_attribute_nodes = []
missing_from_all = []

for mod_name in ast_modules:
    mod = importlib.import_module('pyccel.ast.'+mod_name)
    all_attr = getattr(mod, '__all__', None)
    if all_attr:
        sorted_all = list(all_attr)
        sorted_all.sort()
        if sorted_all != list(all_attr):
            non_alphabetical_all.append(mod_name)
    else:
        missing_all.append(mod_name)

    classes = inspect.getmembers(mod, inspect.isclass)
    for cls_name, cls_obj in classes:
        if inspect.getmodule(cls_obj) is not mod:
            continue
        super_classes = cls_obj.mro()[1:]
        if '__slots__' not in cls_obj.__dict__:
            missing_slots.append(f"{mod_name}.{cls_name}")
        else:
            slots = cls_obj.__slots__
            for c in super_classes:
                if '__slots__' not in c.__dict__:
                    continue
                elif any(s in slots for s in c.__slots__):
                    overridden_slots.append(f'Slot values are overwritten between `{mod_name}.{cls_name}` and `{c.__name__}`')

        if Basic in super_classes:
            if cls_obj not in (PyccelAstNode, ScopedNode) and not isinstance(cls_obj._attribute_nodes, tuple): #pylint: disable=W0212
                missing_attribute_nodes.append(f"{mod_name}.{cls_name}")

        if all_attr and cls_name not in all_attr:
            missing_from_all.append(f"{mod_name}.{cls_name}")

print("Missing __all__")
print(missing_all)
print("__all__ non-alphabetical")
print(non_alphabetical_all)
print("Missing __slots__")
print(missing_slots)
print("Missing _attribute_nodes")
print(missing_attribute_nodes)
print("Not in __all__")
print(missing_from_all)
print("Misused slots")
print(overridden_slots)

with open(args.output, "w", encoding="utf-8") as out:
    # Report error
    if missing_all:
        print("## Missing `__all__`", file=out)
        for f in missing_all:
            print(f"-   `pyccel.ast.{f}`", file=out)
    if non_alphabetical_all:
        print("## Non-alphabetical `__all__`", file=out)
        for f in non_alphabetical_all:
            print(f"-   `pyccel.ast.{f}`", file=out)
    if missing_from_all:
        print("## Classes missing from `__all__`", file=out)
        for f in missing_from_all:
            print(f"-   `pyccel.ast.{f}`", file=out)
    if missing_slots:
        print("## Classes with no `__slots__`", file=out)
        for f in missing_slots:
            print(f"-   `pyccel.ast.{f}`", file=out)
    if missing_attribute_nodes:
        print("## Classes with no `_attribute_nodes`", file=out)
        for f in missing_attribute_nodes:
            print(f"-   `pyccel.ast.{f}`", file=out)
    if overridden_slots:
        print("## Misused slots", file=out)
        for o in overridden_slots:
            print("-  ", o, file=out)

failure = (bool(missing_all) or # bool(non_alphabetical_all) or
          bool(missing_slots) or bool(missing_attribute_nodes) or
          bool(overridden_slots))

if failure:
    sys.exit(1)
else:
    sys.exit(0)
