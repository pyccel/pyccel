""" Script to check that Pyccel coding conventions concerning Pylint are correctly followed
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import json
from git_evaluation_tools import get_diff_as_json

accepted_pylint_commands = {re.compile('.*/IMPORTING_EXISTING_IDENTIFIED3.py'):['reimported'],
                            re.compile('.*/TOO_FEW_ARGS.py'):['no-value-for-parameter'],
                            re.compile('.*/UNKNOWN_IMPORT.py'):['unused-import'],
                            re.compile('.*/UNKNOWN_IMPORT2.py'):['unused-import'],
                            re.compile('.*/USELESS_EXPRESSION.py'):['pointless-statement'],
                            re.compile('tests/errors/known_bugs/dicts.py'):['pointless-statement'],
                            re.compile('.*/syntax/.*'):['pointless-statement','undefined-variable'],
                            re.compile('tests/codegen/fcode/scripts/precision.py'):['unused-variable'],
                            re.compile('tests/semantic/scripts/expressions.py'):['unused-variable'],
                            re.compile('tests/semantic/scripts/calls.py'):['unused-variable']}

def run_pylint(file, flag, messages):
    """
    Check for an expected pylint disable flag.

    Check for an expected pylint disable flag. If the flag is present
    then it is ignored by removing it from the list. Otherwise if the
    file raises the error a message is saved recommending that the
    flag be disabled in the file.

    Parameters
    ----------
    file : file object
        The file being analysed.
    flag : str
        The name of the flag being investigated.
    messages : list
        The list of messages which should be printed.
    """
    with subprocess.Popen([shutil.which('pylint'), file, '--disable=all', f'--enable={flag}']) as r:
        r.communicate()
        result = r.returncode
    if result:
        output_item = {
            "annotations": [],
            "flags":flag,
            "summary":"-  Feel free to disable",
        }
        output_item["annotations"].append({
            "annotation_level":"warning",
            "start_line":1,
            "end_line":1,
            "path":file,
            "message":f"Feel free to disable `{flag}`"
        })
        messages.append(output_item)

def check_expected_pylint_disable(file, disabled, flag, messages, file_changed):
    """
    Check for an expected pylint disable flag.

    Check for an expected pylint disable flag. If the flag is present
    then it is ignored by removing it from the list. Otherwise, if the
    file is modified in this pull request, and it raises the error, a
    message is saved recommending that the flag be disabled in the file.

    Parameters
    ----------
    file : file object
        The file being analysed.
    disabled : list
        The name of all pylint flags disabled in this file.
    flag : str
        The name of the flag being investigated.
    messages : list
        The list of messages which should be printed.
    file_changed : bool
        Indicates whether the file was changed in this diff.
    """
    disabled_copy = disabled.copy()
    if disabled:
        for flags, line_number in disabled_copy:
            if flag in flags:
                new_flags = tuple(value for value in flags if value != flag)
                disabled.remove((flags, line_number))
                if new_flags:
                    disabled.add((new_flags, line_number))
            elif file_changed:
                run_pylint(file, flag, messages)
    elif file_changed:
        run_pylint(file, flag, messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
    parser.add_argument('folder', type=str,
                            help='The folder to be analysed')
    parser.add_argument('diffFile', metavar='diffFile', type=str,
                            help='File containing the git diff output')
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')

    args = parser.parse_args()

    diff = get_diff_as_json(args.diffFile)

    folder = args.folder

    files = [os.path.relpath(os.path.join(root,f)) for root, dirs, filenames in os.walk(folder) for f in filenames if os.path.splitext(f)[1] == '.py']

    success = True

    messages = {"title":"Pyccel_lint","summary":"## Pylint Interaction:\n\n","annotations":[]}

    for f in files:
        with open(f, encoding="utf-8") as myfile:
            lines = [l.replace(' ','') for l in myfile.readlines()]
        pylint_lines_and_numbers = [(l.strip(), i) for i,l in enumerate(lines,1) if l.startswith('#pylint:disable=')]
        disabled = set()
        for value, key in pylint_lines_and_numbers:
            disabled.update([(tuple(value.split('=')[1].split(',')), key)])
        for r,d in accepted_pylint_commands.items():
            if r.match(f):
                for di in d:
                    updated_disabled = disabled.copy()
                    for item in updated_disabled:
                        statements, num = item
                        strings_list = list(statements)
                        if di in strings_list:
                            strings_list.remove(di)
                            disabled.discard(item)
                            if strings_list:
                                disabled.update([(tuple(strings_list), num)])
        file_changed = f in diff
        p = pathlib.Path(f)
        if p.parts[0] == 'tests':
            msg = []
            check_expected_pylint_disable(f, disabled, 'missing-function-docstring', msg, file_changed)
            check_expected_pylint_disable(f, disabled, 'missing-module-docstring', msg, file_changed)
            check_expected_pylint_disable(f, disabled, 'missing-class-docstring', msg, file_changed)
            first_iteration = True
            for item in msg:
                if first_iteration:
                    messages['summary'] += item['summary'] + ' `' + item['flags'] + '`'
                    for value in item['annotations']:
                        messages['annotations'].append(value)
                    first_iteration = False
                else:
                    messages['summary'] += ', `' + item['flags'] + '`'
                    messages['annotations'][-1]['message'] += ', ' + item['flags']
            if not first_iteration:
                messages['summary'] += f' in `{f}`\n\n'
                messages['annotations'][-1]['message'] += f' in {f}'
            if p.parts[1] == 'epyccel':
                updated_disabled = disabled.copy()
                for item in updated_disabled:
                    statements, num = item
                    strings_list = list(statements)
                    if 'reimported' in strings_list:
                        strings_list.remove('reimported')
                        disabled.discard(item)
                        if strings_list:
                            disabled.update([(tuple(strings_list), num)])
        if disabled:
            first_iteration = True
            if file_changed:
                summary_template = f"-  New unexpected pylint disables found in `{f}`: "
                annotation_level = "failure"
                annotation_message = "[ERROR] New unexpected pylint disables: "
            else:
                summary_template = f"-  Unexpected pylint disables found in `{f}`: "
                annotation_level = "warning"
                annotation_message = "Unexpected pylint disables: "
            first_iteration = True
            for value, key in disabled:
                for v in value:
                    if first_iteration:
                        messages['summary'] += f"{summary_template}`{v}`"
                        messages['annotations'].append({
                            'path':f,
                            'start_line':key,
                            'end_line':key,
                            'annotation_level':annotation_level,
                            'message':f"{annotation_message}{v}"})
                        first_iteration = False
                    else:
                        messages['summary'] += ', `' + v + '`'
                        if key == messages['annotations'][-1]['start_line']:
                            messages['annotations'][-1]['message'] += ', ' + v
                        else:
                            messages['annotations'].append({
                                'path':f,
                                'start_line':key,
                                'end_line':key,
                                'annotation_level':annotation_level,
                                'message':f"{annotation_message}{v}"})
            if not first_iteration:
                messages['summary'] += '\n\n'
            success &= (not file_changed)

    if not messages['summary'] and success:
        messages['summary'] = "## Pylint Interaction:\n\n**Success**:The operation was successfully completed. All necessary tasks have been executed without any errors or warnings."
        messages.pop('annotations')
    if not success and not messages['summary']:
        messages['summary'] = "## Pylint Interaction:\n\n**Error**: Something went wrong"
        messages.pop('annotations')

    with open('test_json_result.json', mode='r', encoding="utf-8") as json_file:
        slots_data = json.load(json_file)
        slots_data['summary'] += messages['summary']
        if "annotations" in slots_data:
            slots_data['annotations'].extend(messages['annotations'])
        elif messages['annotations']:
            slots_data.update({'annotations': messages['annotations']})
    with open('test_json_result.json', mode='w', encoding="utf-8") as json_file:
        json.dump(slots_data, json_file)
    with open(args.output, mode='a', encoding="utf-8") as md_file:
        md_file.write(messages['summary'])

    if not success:
        sys.exit(1)
