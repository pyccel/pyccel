""" Script to list the python files changed by the PR
"""
import argparse
import inspect
import os
import sys

parser = argparse.ArgumentParser(description='Collect the files that have been modified by the PR to check their docstrings')
parser.add_argument('gitdiff', metavar='gitdiff', type=str,
                        help='Output of git diff between PR branch and base branch')
parser.add_argument('result', metavar='result', type=str,
                        help='File to save the results')
args = parser.parse_args()

with open(args.gitdiff, encoding="utf-8") as f:
    lines = f.readlines()

lines = [l for l in lines if l.startwith('diff --git a/pyccel')]
lines = [l.split()[3][2:] for l in lines]
with open(args.result, 'w', encoding="utf-8") as f:
    for l in lines:
        print(l, file=f)
