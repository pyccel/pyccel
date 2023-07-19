""" Script to list the objects that should be numpydoc validated
"""
import ast
from _ast import FunctionDef, ClassDef
import argparse
from pathlib import PurePath
from git_evaluation_tools import get_diff_as_json

def should_ignore(name):
    '''
    Determine if an object should be ignored from numpydoc validation.

    Parameters
    ----------
    name : str
        Absolute name of the inspected object.

    Returns
    -------
    bool
        True if object should be ignored, False otherwise.
    '''
    obj_name = name.split('.')[-1]
    #ignore magic methods
    if obj_name.startswith('__') and obj_name.endswith('__'):
        return True
    #ignore _visit_ methods in the SemanticParser class
    if 'SemanticParser._visit_' in name:
        return True
    #ignore _visit_ methods in the SyntaxParser class
    if 'SyntaxParser._visit_' in name:
        return True
    #ignore _print_ methods in the codegen.printing module
    if 'Printer._print_' in name:
        return True
    #ignore _wrap_ methods in the codegen.wrapper module
    if 'Wrapper._wrap_' in name:
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='List the objects with docstrings in the files provided')
    parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                        help='Output of git diff between PR branch and base branch')
    parser.add_argument('output', metavar='output', type=str,
                        help='File to save the output to')
    parser.add_argument('ci_output', metavar='ci_output', type=str,
                        help='File to save the output to for the ci_tools folder')
    args = parser.parse_args()

    results = get_diff_as_json(args.gitdiff)
    pkg_changes = {}
    ci_changes = {}
    for file, upds in results.items():
        filepath = PurePath(file)
        if filepath.parts[0] != 'tests' and filepath.suffix == '.py':
            if filepath.parts[0] == 'ci_tools':
                changes = ci_changes
            else:
                changes = pkg_changes
            for line_no in upds['addition']:
                if file in save_changes:
                    changes[file].append(int(line_no))
                else:
                    changes[file] = [int(line_no)]

    for changes, output in zip([pkg_changes, ci_changes], [args.output, args.ci_output]):
        with open(args.output, 'w', encoding="utf-8") as f:
            for file, line_nos in changes.items():
                with open(file, 'r', encoding="utf-8") as ast_file:
                    tree = ast.parse(ast_file.read())
                # The prefix variable stores the absolute dotted prefix of all
                # objects in the file.
                # for example: the objects in the file pyccel/ast/core.py will
                # have the prefix pyccel.ast.core
                prefix = '.'.join(PurePath(file).with_suffix('').parts)

                objects = []
                to_visit = list(ast.iter_child_nodes(tree))
                for node in to_visit:
                    print(node)
                    # This loop walks the ast and explores all objects
                    # present in the file.
                    # If the object is an instance of a FunctionDef or
                    # a ClassDef, a check is performed to see if any of
                    # the updated lines are present within the object.
                    # Additionally, The name of all objects present
                    # inside a function or a class is updated to include
                    # the name of the parent object
                    if isinstance(node, (FunctionDef, ClassDef)):
                        if should_ignore('.'.join([prefix, node.name])):
                            continue
                        if any((node.lineno <= x <= node.end_lineno
                                for x in line_nos)):
                            objects.append('.'.join([prefix, node.name]))
                        if isinstance(node, ClassDef):
                            obj_pref = node.name
                            for child in node.body:
                                if isinstance(child, (FunctionDef, ClassDef)):
                                    child.name = '.'.join([obj_pref, child.name])
                                    to_visit.append(child)

                for obj in objects:
                    print(obj, file=f)
