""" Look for instances of Python which are not capitalised and report errors
"""
import argparse
import json
from pathlib import Path
import re
import sys
import yaml

from annotation_helpers import locate_code_blocks, is_text, print_to_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    p_args = parser.parse_args()

    files = []

    root = Path(__file__).parent.parent
    for f_pattern in yaml.safe_load(open(root / '.pyspelling.yml'))['matrix'][0]['sources']:
        files.extend(root.glob(f_pattern))

    python_regex = re.compile(r'[^a-zA-Z]python[^a-zA-Z]')

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
                    start = py_match.start()+1
                    end = py_match.end()-1
                    if is_text(l, start, end, line_num, code_blocks):
                        annotations.append({
                            "annotation_level":"failure",
                            "start_line":line_num,
                            "end_line":line_num,
                            "start_column":start,
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
        if f in output:
            print_to_string("", text = output[f])

    # Temporary if to be removed when spelling test outputs errors
    if Path('test_json_result.json').exists():
        with open('test_json_result.json', mode='r', encoding="utf-8") as json_file:
            messages = json.load(json_file)
    else:
        messages = {'summary':''}
    if annotations:
        messages['summary'] += "# Python should be capitalised\n"
        messages.setdefault('annotations', []).extend(annotations)
    for l in output.values():
        text = ''.join(l)
        print(text)
        messages['summary'] += text
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(messages, json_file)

    if annotations:
        sys.exit(1)
