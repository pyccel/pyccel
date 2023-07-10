""" Look for instances of python which are not capitalised and report errors
"""
import argparse
import json
import os
import re

from annotation_helpers import locate_code_blocks, is_text, print_to_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')
    parser.add_argument('files', nargs='*', help='Files to be parsed')
    p_args = parser.parse_args()

    files = p_args.files

    python_regex = re.compile(r'\bpython\b')

    annotations = []
    output = {}
    for f in files:
        with open(f, 'r', encoding='utf-8') as read:
            lines = read.readlines()
        code_blocks = locate_code_blocks(lines)
        for line_num,l in enumerate(lines,1):
            idx = 0
            n = len(l)
            while idx < n:
                py_match = python_regex.search(l, idx)
                if py_match:
                    start = py_match.start()
                    end = py_match.end()
                    if is_text(l, start, end, line_num, code_blocks):
                        annotations.append({
                            "annotation_level":"error",
                            "start_line":line_num,
                            "end_line":line_num,
                            "start_column":start+1,
                            "end_column":end,
                            "path":f,
                            "message": "`python` should be capitalised."
                        })
                        if f not in output:
                            output[f] = []
                            print_to_string(f"Python should be capitalised in the file {f} at the following positions:", text=output[f])
                        print_to_string(f"- Line {line_num}, Columns {start}-{end}" , text = output[f])
                    idx = end
                else:
                    idx = n

    # Temporary if to be removed when spelling test outputs errors
    if os.path.exists('test_json_result.json'):
        with open('test_json_result.json', mode='r', encoding="utf-8") as json_file:
            messages = json.load(json_file)
    else:
        messages = {'summary':''}
    if annotations:
        messages['summary'] += "# Python should be capitalised\n"
        messages.setdefault('annotations', []).extend(annotations)
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(messages, json_file)
    with open(p_args.output, mode='a', encoding="utf-8") as md_file:
        for l in output.values():
            md_file.write(''.join(l))
