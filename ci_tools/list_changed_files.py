""" Script to list the python files changed by the PR
"""
import argparse

parser = argparse.ArgumentParser(
    description='Collect the files that have been modified by the PR to check their docstrings')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                    help='Output of git diff between PR branch and base branch')
parser.add_argument('result', metavar='result', type=str,
                    help='File to save the results')
args = parser.parse_args()

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
