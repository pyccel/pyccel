""" Parse pylint output and format the output for neat bot results.
"""
import argparse
from collections import namedtuple
import json
import sys

import coverage_analysis_tools as cov
from git_evaluation_tools import get_diff_as_json

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
        The dictionary containing the pylint results.
    """
    with open(filename, 'r', encoding='utf-8') as p_file:
        pylint_output = p_file.readlines()

    pylint_output = [l.strip() for l in pylint_output]

    pylint_results = {}
    if not pylint_output:
        return pylint_results

    idx = 0
    line = pylint_output[idx]
    while not all(c=='-' for c in line):
        if not line.startswith('***'):
            file, line, start, _, message = line.split(':', 4)
            pylint_results.setdefault(file, []).append(PylintMessage(file, line, start, message.strip()))
        idx += 1
        line = pylint_output[idx]

    return pylint_results

def filter_pylint_results(pylint_results, diff):
    """
    Filter the pylint results to only show errors relevant to this PR.

    Filter the pylint results to only report errors on lines which have been added
    or changed in this PR.

    Parameters
    ----------
    pylint_results : dict
        The output of get_pylint_results. A dictionary containing the pylint results.
    diff : dict
        The git diff between this branch and the target.

    Returns
    -------
    dict
        A dictionary containing only the pylint errors caused by this branch.
    """
    lines = {k: [int(vi.line) for vi in v] for k,v in pylint_results.items()}
    filtered_lines = cov.compare_coverage_to_diff(lines, diff)

    filtered_pylint_results = {}
    for k,lines in filtered_lines.items():
        orig_errors = pylint_results[k]
        filtered_pylint_results[k] = [v for v in orig_errors if int(v.line) in lines]

    return filtered_pylint_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse pylint output and format the output for neat bot results.')
    parser.add_argument('compare_pylint', type=str,
                            help='The file containing the pylint output from the current branch')
    parser.add_argument('diffFile', metavar='diffFile', type=str,
                            help='File containing the git diff output')
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')
    args = parser.parse_args()

    raw_pylint_results = get_pylint_results(args.compare_pylint)
    diff = get_diff_as_json(args.diffFile)
    filtered_results = filter_pylint_results(raw_pylint_results, diff)

    if filtered_results:
        output = "# Pylint errors found\n"
    else:
        output = "# Success! No pylint errors found\n"

    annotations = []
    for mod, msgs in filtered_results.items():
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
    print(output)

    if annotations:
        sys.exit(1)
