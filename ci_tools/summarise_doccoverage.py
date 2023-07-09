""" Script to check if documentation coverage has decreased
"""
import argparse
import importlib
import inspect
import json
import sys

from list_docs_tovalidate import should_ignore

def print_to_string(*args, text):
    """
    Print to a string and to terminal.

    Wrapper around the printing function to print the same text to both
    the terminal and an output string.

    Parameters
    ----------
    *args : tuple
        Positional arguments to print function.

    *kwargs : dict
        Key word arguments to print function.

    text : list
        A list of strings where the output should also be saved.
    """
    print(*args)
    text.append(' '.join(args)+'\n')

parser = argparse.ArgumentParser(description='Check doc coverage change')
parser.add_argument('base', metavar='head_cov', type=str,
                        help='File containing the coverage of the head branch')
parser.add_argument('compare', metavar='base_cov', type=str,
                        help='File containing the coverage of the base branch')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

results = {}
for branch_file in [args.base, args.compare]:
    branch = branch_file[:-4]
    with open(branch_file, encoding="utf-8") as f:
        lines = f.readlines()

    results[branch + '_summary'] = ''.join(lines[-3:])
    results[branch + '_no_mod'] = set()
    results[branch + '_no_obj'] = {}
    lines = [l for l in lines if l.startswith('File: ') or l.startswith(' - ')]
    n = len(lines)
    i = 0
    while i < n:
        #modname = lines[i].split()[1].strip('"')[:-3].replace('/','.').split(f'.{branch}.', 1)[1]
        modname = lines[i].split()[1].strip('"')[:-3].replace('/','.').split(f'.pyccel.', 1)[1]
        i+=1
        while i<n and lines[i].startswith(' - '):
            if lines[i].startswith(' - No module docstring'):
                results[branch + '_no_mod'].add(modname)
            else:
                objname = lines[i].split()[-1].strip('`')
                if not should_ignore(objname):
                    obj_name_parts = objname.split('.')
                    mod_dict = results[branch + '_no_obj'].setdefault(modname, {})
                    mod_dict.setdefault(obj_name_parts[0], []).extend(obj_name_parts[1:])
            i += 1

added_mod = [mod for mod in results['compare_no_mod'] if mod not in results['base_no_mod']]
added_obj = {(mod, cls): methods for mod, obj in results['compare_no_obj'].items() \
                                 for cls, methods in obj.items() \
                                 if methods != results['base_no_obj'].get(mod, {}).get(cls, None)}

if len(added_mod) > 0 or len(added_obj) > 0:
    annotations = []
    summary = []
    print_to_string('## Failure: Coverage has decreased!', text=summary)
    print_to_string('### Base Branch Summary', text=summary)
    print_to_string(results['base_summary'], text=summary)
    print_to_string('Compare Branch Summary', text=summary)
    print_to_string(results['compare_summary'], text=summary)
    if len(added_mod) > 0:
        print_to_string('### This pull request added these modules without docstrings:', text=summary)
        for idx, mod in enumerate(added_mod):
            print_to_string(f'{idx + 1}. {mod}', text=summary)
            annotations.append({
                "annotation_level":"error",
                "start_line":1,
                "end_line":1,
                "path":mod.replace('.','/')+'.py',
                "message":f"Missing module docstring."
            })
        print_to_string()
    if len(added_obj) > 0:
        print_to_string('### This pull request added these objects without docstrings:', text=summary)
        idx = 0
        for (mod, cls), objects in added_obj.items():
            mod_obj = importlib.import_module(mod)
            for obj in objects:
                method = getattr(getattr(mod_obj, cls), obj)
                # Unpack property to method
                method = getattr(method, 'fget', method)
                source, start_line = inspect.getsourcelines(method)
                length = len(source)
                if obj in results['base_no_obj'].get(mod, {}).get(cls, []):
                    level = 'warning'
                else:
                    level = 'error'
                    print_to_string(f'{idx + 1}.  {mod}.{cls}.{obj}', text=summary)
                    idx += 1
                annotations.append({
                    "annotation_level":level,
                    "start_line":start_line,
                    "end_line":start_line+length-1,
                    "path":mod.replace('.','/')+".py",
                    "message":f"Missing docstring."
                })
            if len(objects) == 0:
                print_to_string(f'{idx + 1}.  {mod}.{cls}', text=summary)
                idx += 1
                method = getattr(mod_obj, cls)
                # Unpack property to method
                method = getattr(method, 'fget', method)
                source, start_line = inspect.getsourcelines(method)
                annotations.append({
                    "annotation_level":"error",
                    "start_line":start_line,
                    "end_line":start_line,
                    "path":mod.replace('.','/')+".py",
                    "message":f"Missing docstring."
                })
        print_to_string(text=summary)
    summary_text = "\n".join(summary)
    messages = {'summary' : summary_text,
                'annotations': annotations}
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(messages, json_file)
    with open(args.output, "w", encoding="utf-8") as out:
        print(summary_text, file=out)

    sys.exit(1)

else:
    with open(args.output, "w", encoding="utf-8") as out:
        print_to_string('# Part 1:', file=out)
        print_to_string('## Success!', file=out)
        print_to_string('### Base Branch Summary', file=out)
        print_to_string(results['base_summary'], file=out)
        print_to_string('### Compare Branch Summary', file=out)
        print_to_string(results['compare_summary'], file=out)
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump({'summary':"Documentation coverage is complete!"}, json_file)
