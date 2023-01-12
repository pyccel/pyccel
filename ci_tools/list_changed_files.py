""" Script to list the python files and lines changed by the PR
"""
import argparse
from git_evaluation_tools import get_diff_as_json as gdj

parser = argparse.ArgumentParser(
    description='Collect the files and lines that have been modified by the PR to check their docstrings')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                    help='Output of git diff between PR branch and base branch')
parser.add_argument('result', metavar='result', type=str,
                    help='File to save the results')
args = parser.parse_args()
results = gdj(args.gitdiff)
with open(args.result,'w', encoding='utf-8') as out:
    for file,changes in results:
        for line in changes['addition']:
            print(file, line, sep=' ', file=out)
        for line in changes['deletion']:
            print(file, line, sep=' ', file=out)
