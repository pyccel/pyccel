""" Script to check if documentation coverage has decreased
"""
import argparse
import sys

parser = argparse.ArgumentParser(description='Check doc coverage change')
parser.add_argument('base', metavar='head_cov', type=str,
                        help='File containing the coverage of the head branch')
parser.add_argument('compare', metavar='base_cov', type=str,
                        help='File containing the coverage of the base branch')

args = parser.parse_args()

results = {}
for branch_file in [args.base, args.compare]:
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
        while i<n and lines[i].startswith(' - '):
            if lines[i].startswith(' - No module docstring'):
                results[branch + '_no_mod'].update([modname])
            else:
                objname = lines[i].split()[-1].strip('`')
                results[branch + '_no_obj'].update([objname])
            i += 1

added_mod = [mod for mod in results['compare_no_mod'] if mod not in results['base_no_mod']]
added_obj = [obj for obj in results['compare_no_obj'] if obj not in results['base_no_obj']]

if len(added_mod) > 0 or len(added_obj) > 0:
    print('Failure: Coverage has decreased!')
    print('Base Branch Summary')
    print(results['base_summary'])
    print('Compare Branch Summary')
    print(results['compare_summary'])
    if len(added_mod) > 0:
        print('This pull request added these modules without docstrings:')
        for mod in added_mod:
            print(f'\t * {mod}')
        print()
    if len(added_obj) > 0:
        print('This pull request added these objects without docstrings:')
        for obj in added_obj:
            print(f'\t * {obj}')
        print()
    sys.exit(1)
else:
    print('Success!')
    print('Base Branch Summary')
    print(results['base_summary'])
    print('Compare Branch Summary')
    print(results['compare_summary'])
    sys.exit(0)