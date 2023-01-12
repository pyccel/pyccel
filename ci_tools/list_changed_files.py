""" Script to list the python files and lines changed by the PR
"""
import argparse
from git_evaluation_tools import git_diff_as_json as gdj

parser = argparse.ArgumentParser(
    description='Collect the files and lines that have been modified by the PR to check their docstrings')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                    help='Output of git diff between PR branch and base branch')
parser.add_argument('result', metavar='result', type=str,
                    help='File to save the results')
args = parser.parse_args()
<<<<<<< HEAD
results = gdj(args.gitdiff)
with open(args.result,'w', encoding='utf-8') as out:
    for file,changes in results:
        for line in changes['addition']:
            print(file, line, sep=' ', file=out)
        for line in changes['deletion']:
            print(file, line, sep=' ', file=out)
=======

with open(args.gitdiff, encoding="utf-8") as f:
    lines = f.readlines()
changes = []
for idx, l in enumerate(lines):
    if l.startswith('diff --git a/pyccel') and \
        not lines[idx + 1].startswith('deleted file '):
        changes.append(l)
changes = [l.split()[3][2:] for l in changes]
with open(args.result, 'w', encoding="utf-8") as f:
    for l in changes:
        print(l, file=f)
>>>>>>> 86324854d98bd95094345b2221bb1cddad675d59
