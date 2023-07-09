""" Script to check if documentation coverage has decreased
"""
import argparse
import json
import os
import sys

from list_docs_tovalidate import should_ignore
from annotation_helpers import print_to_string, get_code_file_and_lines

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
        modname = lines[i].split()[1].strip('"')[:-3].replace('/','.').split(f'.{branch}.', 1)[1]
        i+=1
        while i<n and lines[i].startswith(' - '):
            if lines[i].startswith(' - No module docstring'):
                results[branch + '_no_mod'].add(modname)
            else:
                objname = lines[i].split()[-1].strip('`')
                if not should_ignore(objname):
                    obj_name_parts = objname.split('.')
                    mod_dict = results[branch + '_no_obj'].setdefault(modname, {})
                    mod_dict.setdefault(obj_name_parts[0], []).append(obj_name_parts[1:])
            i += 1

added_mod = [mod for mod in results['compare_no_mod'] if mod not in results['base_no_mod']]
added_obj = {(mod, cls): methods for mod, obj in results['compare_no_obj'].items() \
                                 for cls, methods in obj.items() \
                                 if methods != results['base_no_obj'].get(mod, {}).get(cls, None)}

base_folder = os.path.abspath(args.base[:-4])

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
                "message":"Missing module docstring."
            })
        print_to_string(text=summary)
    if len(added_obj) > 0:
        print_to_string('### This pull request added these objects without docstrings:', text=summary)
        idx = 0
        for (mod, cls), objects in added_obj.items():
            if [] in objects:
                file, start, end = get_code_file_and_lines(cls, base_folder, mod)
                print_to_string(f'{idx + 1}.  {mod}.{cls}', text=summary)
                idx += 1
                annotations.append({
                    "annotation_level":"error",
                    "start_line":start,
                    "end_line":end,
                    "path":file,
                    "message":"Missing docstring."
                })
            for obj in objects:
                if obj == []:
                    continue
                obj_name = '.'.join(obj)
                file, start, end = get_code_file_and_lines(f"{cls}.{obj_name}", base_folder, mod)
                if obj in results['base_no_obj'].get(mod, {}).get(cls, []):
                    level = 'warning'
                else:
                    level = 'error'
                    print_to_string(f'{idx + 1}.  {mod}.{cls}.{obj_name}', text=summary)
                    idx += 1
                annotations.append({
                    "annotation_level":level,
                    "start_line":start,
                    "end_line":end,
                    "path":file,
                    "message":"Missing docstring."
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
    summary = []
    print_to_string('# Part 1:', text=summary)
    print_to_string('## Success!', text=summary)
    print_to_string('### Base Branch Summary', text=summary)
    print_to_string(results['base_summary'], text=summary)
    print_to_string('### Compare Branch Summary', text=summary)
    print_to_string(results['compare_summary'], text=summary)
    summary_text = "\n".join(summary)
    with open(args.output, "w", encoding="utf-8") as out:
        print(summary_text, file=out)
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump({'summary':"# Documentation coverage is complete!"}, json_file)
