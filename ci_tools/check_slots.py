""" Script to check that Pyccel coding conventions concerning slots are correctly followed in the AST
"""
import argparse
import importlib
import inspect
import pathlib
import sys
import json
from pyccel import ast
from pyccel.ast.basic import PyccelAstNode, TypedAstNode, ScopedAstNode

def extract_dict_elements(input_dict):
    """
    Write a dictionary as a dictionary compatible with GitHub's check runs output.

    Extract specific elements from the input dictionary based on the given keys and returns a new dictionary.

    Parameters
    ----------
    input_dict : dict
        The input dictionary from which elements will be extracted.

    Returns
    -------
    dict
        A new dictionary containing the extracted elements.
    """
    output_dict = {'title':"Pyccel_lint", 'summary':"## Check Slots:\n\n",'annotations':[]}
    for values in input_dict.items():
        value_list = values[-1]
        if value_list:
            output_dict['summary'] += f"### {value_list[0]['title']} :\n\n"
            output_dict['summary'] += "\n\n".join(f"-  {value['message']}" for value in value_list)
            output_dict['summary'] += "\n\n"
            output_dict['annotations'].extend(value['annotations'] for value in value_list)
    return output_dict

def fill_dictionary(title, message, file, start_line, end_line, annotation_level, annotation_msg):
    """
    Fill a dictionary with the provided data.

    Fill a dictionary with the provided data to describe an annotation on a GitHub
    check run.

    Parameters
    ----------
    title : str
        The title of the check run output.

    message : str
        A description of the output.

    file : str
        The file to be annotated.

    start_line : int
        The line at the start of the annotation.

    end_line : int
        The line at the end of the annotation.

    annotation_level : str
        The level of the anntoation [failure/warning].

    annotation_msg : str
        The message in the annotation.

    Returns
    -------
    dict
        A dictionary filled with the provided data.
    """
    index = file.find("site-packages")
    filled_dict = {
        'title':title,
        'message':f"`{message}`",
        'annotations':{
            'path':file[index + len("site-packages") + 1:],
            'start_line':start_line,
            'end_line':end_line,
            'annotation_level':annotation_level,
            'message':annotation_msg
        }
    }
    return filled_dict

def sort_key(name : str):
    """
    A method to split a string into numeric and non-numeric sections for improved sorting.

    A method to split a string into numeric and non-numeric sections for improved sorting.

    Parameters
    ----------
    name : str
        The string from the list being sorted.

    Results
    -------
    tuple[str|int]
        The key by which the string should be sorted.
    """
    sections = []
    n = len(name)
    while n > 0:
        if name[0].isdigit():
            i = next((i for i,s in enumerate(name) if not s.isdigit()), n)
            sections.append(int(name[:i]))
        else:
            i = next((i for i,s in enumerate(name) if s.isdigit()), n)
            sections.append(name[:i])
        n -= i
    return tuple(sections)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
    args = parser.parse_args()

    # Get ast modules
    ast_folder = pathlib.Path(ast.__file__).parent
    ast_modules = ['.'.join(f.relative_to(ast_folder).parts)[:-3] for f in ast_folder.rglob('*.py') if f.parts[-1] != '__init__.py']

    # Prepare error collection
    missing_all = []
    non_alphabetical_all = []
    missing_slots = []
    overridden_slots = []
    missing_attribute_nodes = []
    missing_from_all = []

    error_collection = {
        'missing_all':[],
        'badly_grouped_all': [],
        'bad_all_group': [],
        'non_alphabetical_all':[],
        'missing_slots':[],
        'overridden_slots':[],
        'missing_attribute_nodes':[],
        'missing_from_all':[]
    }

    for mod_name in ast_modules:
        mod = importlib.import_module('pyccel.ast.'+mod_name)
        all_attr = getattr(mod, '__all__', None)
        if all_attr:
            sorted_all = list(all_attr)
            sorted_all.sort(key=sort_key)
            # If not already sorted
            if sorted_all != list(all_attr):
                # Get relevant lines
                lines = inspect.getsource(mod).splitlines()
                start_line = -1
                end_line = -1
                for line_num, line in enumerate(lines):
                    if '__all__' in line:
                        start_line = line_num
                        while ')' not in line:
                            line_num += 1
                            line = lines[line_num]
                        end_line = line_num + 1
                        break

                # Get all tuple elements including comments
                all_code = '\n'.join(lines[start_line:end_line])
                all_keys = [li.strip(' ,\'"') for l in all_code.strip().rstrip(')').lstrip('__all__').strip().lstrip('= (').split(',') \
                                for li in l.split('\n') if li and not li.isspace()]

                # Split unsorted keys into groups according to comments
                groups = {l.strip('# -'): [] for l in all_keys if l.startswith('#')}
                if len(set(groups)) < len(groups):
                    error_collection['badly_grouped_all'].append(fill_dictionary("`__all__` split into multiple groups with the same name", f"pyccel.ast.{mod_name}",
                        inspect.getfile(mod), start_line, end_line, "failure", f"Fix the groups in the __all__ attribute of `{mod_name}`"))
                if any(len(g) == 0 for g in groups):
                    error_collection['bad_all_group'].append(fill_dictionary("`__all__` is split into unlabelled groups", f"pyccel.ast.{mod_name}",
                        inspect.getfile(mod), start_line, end_line, "failure", f"Fix the groups in the __all__ attribute of `{mod_name}`"))

                if not all_keys[0].startswith('#'):
                    groups['start'] = []
                    current_group = 'start'

                for l in all_keys:
                    if l.startswith('#'):
                        current_group = l.strip('# -')
                    else:
                        groups[current_group].append(l)

                # Check if keys are sorted within each group
                for n, g in groups.items():
                    o_g = list(g)
                    g.sort(key=sort_key)
                    if g != o_g:
                        print(g)
                        print(o_g)
                        name = f"pyccel.ast.{mod_name}"
                        if n != 'start':
                            name += f'[{n}]'
                        error_collection['non_alphabetical_all'].append(fill_dictionary("Non-alphabetical `__all__`", name,
                            inspect.getfile(mod), start_line, end_line, "failure", f"Sort the __all__ attribute of `{mod_name}`"))
        else:
            error_collection['missing_all'].append(fill_dictionary("Missing `__all__`", f"pyccel.ast.{mod_name}",
                inspect.getfile(mod), 1, 1, "failure", f"Missing __all__ attribute in: `{mod_name}`"))

        classes = inspect.getmembers(mod, inspect.isclass)
        for cls_name, cls_obj in classes:
            if inspect.getmodule(cls_obj) is not mod:
                continue
            super_classes = cls_obj.mro()[1:]
            if '__slots__' not in cls_obj.__dict__:
                sourceline = inspect.getsourcelines(cls_obj)[1]
                error_collection['missing_slots'].append(fill_dictionary("Classes with no `__slots__`", f"pyccel.ast.{mod_name}.{cls_name}",
                    inspect.getfile(mod), sourceline, sourceline, "failure", f"`{mod_name}.{cls_name}` class with no `__slots__`"))
            else:
                slots = cls_obj.__slots__
                for c in super_classes:
                    if '__slots__' not in c.__dict__:
                        continue
                    elif any(s in slots for s in c.__slots__):
                        sourceline = inspect.getsourcelines(cls_obj)[1]
                        error_collection['overridden_slots'].append(fill_dictionary("Overwritten slot values", f"pyccel.ast.{mod_name}.{cls_name}",
                            inspect.getfile(mod), sourceline, sourceline, "failure", f"Slot values are overwritten between `{mod_name}.{cls_name}` and `{c.__name__}`"))

            if PyccelAstNode in super_classes:
                if cls_obj not in (TypedAstNode, ScopedAstNode) and not isinstance(cls_obj._attribute_nodes, tuple): #pylint: disable=W0212
                    sourceline = inspect.getsourcelines(cls_obj)[1]
                    error_collection['missing_attribute_nodes'].append(fill_dictionary("Classes with no `_attribute_nodes`", f"pyccel.ast.{mod_name}.{cls_name}",
                            inspect.getfile(mod), sourceline, sourceline, "failure", f"Missing attribute nodes in : `{mod_name}.{cls_name}`"))

            if all_attr and cls_name not in all_attr:
                sourceline = inspect.getsourcelines(cls_obj)[1]
                error_collection['missing_from_all'].append(fill_dictionary("Classes missing from `__all__`", f"pyccel.ast.{mod_name}.{cls_name}",
                    inspect.getfile(mod), sourceline, sourceline, "failure", f"`{mod_name}.{cls_name}` is missing from `__all__`"))

    messages = extract_dict_elements(error_collection)
    if not messages['annotations']:
        messages['summary'] = "Check Slots\n\n**Success**:The operation was successfully completed. All necessary tasks have been executed without any errors or warnings.\n\n"
        messages.pop('annotations')
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(messages, json_file)

    # Report error
    print("# " + messages['title'] + '\n')
    print(messages['summary'])

    failure = (bool(error_collection['missing_all']) or # bool(error_collection['non_alphabetical_all']) or
              bool(error_collection['missing_slots']) or bool(error_collection['missing_attribute_nodes']) or
              bool(error_collection['overridden_slots']) or bool(error_collection['missing_from_all']))


    if failure:
        sys.exit(1)
    else:
        sys.exit(0)
