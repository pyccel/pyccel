""" Script to check if documentation coverage has decreased
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Check doc coverage change')
parser.add_argument('head', metavar='head_cov', type=str,
                        help='File containing the coverage of the head branch')
parser.add_argument('base', metavar='base_cov', type=str,
                        help='File containing the coverage of the base branch')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the report will be printed')

args = parser.parse_args()

results = {}
for branch_file in [args.head, args.base]:
    branch = branch_file[:-4]
    with open(branch_file, encoding="utf-8") as f:
        lines = f.readlines()

    results[branch + '_summary'] = ''.join(lines[-3:])
    results[branch + '_no_mod'] = set()
    results[branch + '_no_obj'] = set()
    lines = [l for l in lines if l.startswith('File: ') or l.startswith(' - ')]
    n = len(lines)
    i = 0
    while i < n:
        modname = lines[i].split()[1].strip('"')[:-3].replace('/','.').split(branch, 1)[1][1:]
        i+=1
        words = set()
        while i<n and lines[i].startswith(' - '):
            if lines[i].startswith(' - No module docstring'):
                results[branch + '_no_mod'].update(modname)
            else:
                objname = lines[i].split()[-1].strip('`')
                results[branch + '_no_obj'].update(objname)
            i += 1

added_mod = [mod for mod in results['base_no_mod'] if mod not in results['head_no_mod']]
added_obj = [obj for obj in results['base_no_obj'] if obj not in results['head_no_obj']]
with open(args.output, 'w', encoding="utf-8") as f:
    print('Base Branch Summary', file=f)
    print(results['base_summary'], file=f)
    print('Head Branch Summary', file=f)
    print(results['head_summary'], file=f)
    if len(added_mod) > 0:
        print('This pull request added these modules without docstrings:', file=f)
        for mod in added_mod:
            print(f'\t * {mod}', file=f)
        print(file=f)
    if len(added_obj) > 0:
        print('This pull request added these objects without docstrings:', file=f)
        for obj in added_obj:
            print(f'\t * {obj}', file=f)
        print(file=f)
'''        
if len(added_mod) > 0 or len(added_obj) > 0:
    sys.exit(1)
else:
    sys.exit(0)
'''