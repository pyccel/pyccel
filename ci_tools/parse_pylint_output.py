""" Parse pylint output and format the output for neat bot results.
"""
import argparse
from collections import namedtuple
import json
import sys

PylintMessage = namedtuple('PylintMessage', ['file','line', 'position', 'message'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse pylint output and format the output for neat bot results.')
    parser.add_argument('pylint', type=str,
                            help='The file containing the pylint output')
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')
    args = parser.parse_args()

    with open(args.pylint, 'r', encoding='utf-8') as p_file:
        pylint_output = p_file.readlines()

    pylint_output = [l.strip() for l in pylint_output]

    pylint_results = {}
    idx = 0
    line = pylint_output[idx]
    while not all(c=='-' for c in line):
        if line.startswith('***'):
            _, key = line.split(' Module ')
            pylint_results[key] = []
        else:
            file, line, start, code, message = line.split(':')
            pylint_results[key].append(PylintMessage(file, line, start, message.strip()))
        idx += 1
        line = pylint_output[idx]

    if pylint_results:
        output = "# Pylint errors found\n"
    else:
        output = "# Success! No pylint errors found\n"

    annotations = []
    for mod, msgs in pylint_results.items():
        output += f"## Errors found in module {mod}\n"
        for m in msgs:
            output += f"-  On line {m.line} : {m.message}\n"
            annotations.append({
                                'path':m.file,
                                'start_line':int(m.line),
                                'end_line':int(m.line),
                                'start_column':int(m.position),
                                'annotation_level':'failure',
                                'message':m.message})

    json_data = {'summary': output}
    if annotations:
        json_data['annotations'] = annotations

    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(json_data, json_file)
    with open(args.output, mode='a', encoding="utf-8") as md_file:
        md_file.write(output)

    if annotations:
        sys.exit(1)
