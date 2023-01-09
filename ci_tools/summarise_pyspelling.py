""" Script to output pyspelling results in a more digestible markdown format
"""
import argparse
import difflib
import os
import sys

parser = argparse.ArgumentParser(description='Create a neat markdown file to summarise the results')
parser.add_argument('spelling', metavar='diffFile', type=str,
                        help='File containing the pyspelling output')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

with open(args.spelling, encoding="utf-8") as f:
    lines = f.readlines()

lines = [l.strip() for l in lines[:-1]]
lines = [l for l in lines if l != 'Misspelled words:' and any(c != '-' for c in l)]

errors = {}

n = len(lines)
i = 0
while i < n:
    filename = lines[i].split()[1]
    i+=1
    words = set()
    while i<n and not lines[i].startswith('<htmlcontent>'):
        words.add(lines[i])
        i+=1

    if filename in errors:
        errors[filename].update(words)
    else:
        errors[filename] = words

if errors:
    all_words = set()
    with open(args.output, 'w', encoding="utf-8") as f:
        print("There are misspelled words", file=f)
        for name, words in errors.items():
            print("## `", name, "`", file=f)
            for w in words:
                suggestions = difflib.get_close_matches(word, internal_dict)
                if suggestions:
                    print("-   ", w, "  f:  Did you mean {word} -> {suggestions}", file=f)
                else:
                    print("-   ", w, file=f)
            print(file=f)
            all_words.update(words)

        print("These errors may be due to typos, capitalisation errors, or lack of quotes around code. If this is a false positive please add your word to `.dict_custom.txt`", file=f)

        with open(os.path.join(os.path.dirname(__file__),'..','.dict_custom.txt'), encoding="utf-8") as d:
            internal_dict = [w.strip() for w in d.readlines()]

    sys.exit(1)
else:
    sys.exit(0)
