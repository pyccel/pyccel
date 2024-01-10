""" Parse pylint output and format the output for neat bot results.
"""
import argparse
from collections import namedtuple
import json
import sys

PylintMessage = namedtuple('PylintMessage', ['file','line', 'position', 'message'])

def get_pylint_results(filename):
    """
    Extract pylint results from a file.

    Parse a file containing pylint results and save the results
    to a dictionary whose keys are modules and whose values are
    PylintMessage objects.

    Parameters
    ----------
    filename : str
        The name of the file being parsed.

    Returns
    -------
    dict
        The dictionary containg the pylint results.
    """
    with open(filename, 'r', encoding='utf-8') as p_file:
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
            file, line, start, code, message = line.split(':', 4)
            pylint_results[key].append(PylintMessage(file, line, start, message.strip()))
        idx += 1
        line = pylint_output[idx]

    return pylint_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse pylint output and format the output for neat bot results.')
    parser.add_argument('base_pylint', type=str,
                            help='The file containing the pylint output from the devel branch')
    parser.add_argument('compare_pylint', type=str,
                            help='The file containing the pylint output from the current branch')
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')
    args = parser.parse_args()

    base_pylint_results = get_pylint_results(args.base_pylint)
    compare_pylint_results = get_pylint_results(args.compare_pylint)

    pylint_results = {}
    for k,v in compare_pylint_results.items():
        if k not in base_pylint_results:
            pylint_results[k] = v
        else:
            base_v = base_pylint_results[k]
            new_messages = [vi for vi in v if vi not in base_v]
            if new_messages:
                pylint_results[k] = new_messages

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
