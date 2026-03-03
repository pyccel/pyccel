"""Check that the PR description meets the minimum requirements.

Checks:
1. The first non-empty line is not the boilerplate placeholder.
2. The first non-empty line contains more than 2 words.
3. All checklist items are ticked (no unchecked '- [ ]' boxes remain).
"""
import os
import pathlib
import sys

root = pathlib.Path(__file__).parent.parent

with open(root / '.github' / 'pull_request_template.md', 'r', encoding='utf-8') as f:
    original_body = f.readlines()

body = os.environ.get('PR_BODY', '')
lines = body.splitlines()

first_line = next((l for l in lines if l.strip()), '')

if first_line.strip() == original_body[0].strip():
    print("ERROR: The PR description still contains the boilerplate placeholder. "
          "Please replace it with a description of your changes.")
    sys.exit(1)

split_index = next(i for i,l in enumerate(lines) if l.strip() == '---', len(lines))
pr_description = '\n'.join(l for l in lines[:split_index]).strip()

if len(pr_description.split()) <= 2:
    print("ERROR: The PR description is too short. "
          "Please provide a meaningful description of more than 2 words.")
    sys.exit(1)

if '- [ ]' in body:
    print("ERROR: Not all checklist items have been ticked. "
          "Please complete the PR checklist before marking the PR as ready.")
    sys.exit(1)

if body.count('- [x]') != sum(1 if line.startswith('- [ ]') else 0 for line in original_body):
    print("ERROR: Missing checklist items. "
          "Please mark irrelevant items as complete instead of deleting them to help reviewers.")
    sys.exit(1)

print("PR description checks passed.")
