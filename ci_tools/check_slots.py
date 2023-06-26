""" Script to check that Pyccel coding conventions concerning slots are correctly followed in the AST
"""
import argparse
import importlib
import inspect
import os
import sys
import json
import copy
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

ErrorCollection = {
    'missing_all':[],
    'non_alphabetical_all':[],
    'missing_slots':[],
    'overridden_slots':[],
    'missing_attribute_nodes':[],
    'missing_from_all':[]
}

def extract_dict_elements(input_dict):
    """
    Extracts specific elements from the input dictionary based on the given keys and returns a new dictionary.

    Args:
        input_dict (dict): The input dictionary from which elements will be extracted.

    Returns:
        dict: A new dictionary containing the extracted elements.
    """
    output_dict = {'title':"Check Slots", 'summary':"",'annotations':[]}
    for values in input_dict.items():
        first_iteration = True
        for value in values[-1]:
            if value:
                if first_iteration:
                    output_dict['summary'] += f"{value['title']} :\n"
                    first_iteration = False
                output_dict['summary'] += f"\t-{value['message']}\n"
                output_dict['annotations'].append(value['annotations'])
    return output_dict

def fill_dictionary(title, message, file, start_line, end_line, annotation_level, annotation_msg):
    """
    Fills a dictionary with the provided data.

    Args:
        data (dict): The data to populate the dictionary. It should be a dictionary
                     where the keys represent the dictionary keys, and the values
                     represent the corresponding values.

    Returns:
        dict: A dictionary filled with the provided data.
    """
    filled_dict = {
        'title':title,
        'message':message,
        'annotations':{
            'path':file,
            'start_line':start_line,
            'end_line':end_line,
            'annotation_level':annotation_level,
            'message':annotation_msg
        }
    }
    return filled_dict

for mod_name in ast_modules:
    mod = importlib.import_module('pyccel.ast.'+mod_name)
    all_attr = getattr(mod, '__all__', None)
    if all_attr:
        sorted_all = list(all_attr)
        sorted_all.sort()
        if sorted_all != list(all_attr):
            lines = inspect.getsource(mod).splitlines()
            start_line = -1
            end_line = -1
            for line_num, line in enumerate(lines):
                if '__all__' in line:
                    start_line = line_num + 1
                    while ')' not in line:
                        line_num += 1
                        line = lines[line_num]
                        end_line = line_num + 1
                    ErrorCollection['non_alphabetical_all'].append(fill_dictionary("Non-alphabetical `__all__`", f"`pyccel.ast.{mod_name}`",
                        inspect.getfile(mod), start_line, end_line, "warning", f"Sort the __all__ attribute of `{mod_name}`"))
                    break
    else:
        ErrorCollection['missing_all'].append(fill_dictionary("Missing `__all__`", f"`pyccel.ast.{mod_name}`",
            inspect.getfile(mod), 1, 1, "failure", f"Missing __all__ attribute in: `{mod_name}`"))

    classes = inspect.getmembers(mod, inspect.isclass)
    first_iteration = True
    for cls_name, cls_obj in classes:
        if inspect.getmodule(cls_obj) is not mod:
            continue
        super_classes = cls_obj.mro()[1:]
        if '__slots__' not in cls_obj.__dict__:
            ErrorCollection['missing_slots'].append(fill_dictionary("Classes with no `__slots__`", f"`pyccel.ast.{mod_name}.{cls_name}`",
                inspect.getfile(mod), inspect.getsourcelines(cls_obj)[1], inspect.getsourcelines(cls_obj)[1], "failure", f"`{mod_name}.{cls_name}` classe with no `__slots__`"))
        else:
            slots = cls_obj.__slots__
            for c in super_classes:
                if '__slots__' not in c.__dict__:
                    continue
                elif any(s in slots for s in c.__slots__):
                    ErrorCollection['overridden_slots'].append(fill_dictionary("Overwritten slot values", f"`pyccel.ast.{mod_name}.{cls_name}`",
                        inspect.getfile(mod), inspect.getsourcelines(cls_obj)[1], inspect.getsourcelines(cls_obj)[1], "failure", f"Slot values are overwritten between `{mod_name}.{cls_name}` and `{c.__name__}`"))

        if Basic in super_classes:
            if cls_obj not in (PyccelAstNode, ScopedNode) and not isinstance(cls_obj._attribute_nodes, tuple): #pylint: disable=W0212
                ErrorCollection['missing_attribute_nodes'].append(fill_dictionary("Classes with no `_attribute_nodes`", f"`pyccel.ast.{mod_name}.{cls_name}`",
                        inspect.getfile(mod), inspect.getsourcelines(cls_obj)[1], inspect.getsourcelines(cls_obj)[1], "failure", f"Missing attribute nodes in : `{mod_name}.{cls_name}`"))

        if all_attr and cls_name not in all_attr:
            ErrorCollection['missing_from_all'].append(fill_dictionary("Classes missing from `__all__`", f"`pyccel.ast.{mod_name}.{cls_name}`",
                inspect.getfile(mod), inspect.getsourcelines(cls_obj)[1], inspect.getsourcelines(cls_obj)[1], "failure", f"`{mod_name}.{cls_name}` classe is missing from `__all__`"))

messages = extract_dict_elements(ErrorCollection)
json_data = json.dumps(messages)
with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
    json_file.write(json_data)

with open(args.output, "w", encoding="utf-8") as md_file:
    # Report error
    md_file.write("# " + messages['title'] + '\n\n')
    md_file.write(messages['summary'])

failure = (bool(ErrorCollection['missing_all']) or # bool(ErrorCollection['non_alphabetical_all']) or
          bool(ErrorCollection['missing_slots']) or bool(ErrorCollection['missing_attribute_nodes']) or
          bool(ErrorCollection['overridden_slots']))


if failure:
    sys.exit(1)
else:
    sys.exit(0)
