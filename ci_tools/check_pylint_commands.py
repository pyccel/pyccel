""" Script to check that Pyccel coding conventions concerning Pylint are correctly followed
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
from git_evaluation_tools import get_diff_as_json

accepted_pylint_commands = {re.compile('.*/IMPORTING_EXISTING_IDENTIFIED3.py'):['reimported'],
                            re.compile('.*/TOO_FEW_ARGS.py'):['no-value-for-parameter'],
                            re.compile('.*/UNKNOWN_IMPORT.py'):['unused-import'],
                            re.compile('.*/UNKNOWN_IMPORT2.py'):['unused-import'],
                            re.compile('.*/USELESS_EXPRESSION.py'):['pointless-statement'],
                            re.compile('./tests/errors/known_bugs/dicts.py'):['pointless-statement'],
                            re.compile('.*/syntax/.*'):['pointless-statement','undefined-variable'],
                            re.compile('./tests/codegen/fcode/scripts/precision.py'):['unused-variable'],
                            re.compile('./tests/semantic/scripts/expressions.py'):['unused-variable'],
                            re.compile('./tests/semantic/scripts/calls.py'):['unused-variable']}

def check_expected_pylint_disable(file, disabled, flag, messages):
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
    disabled : list
        The name of all pylint flags disabled in this file.
    flag : str
        The name of the flag being investigated.
    messages : list
        The list of messages which should be printed.
    """
    if flag in disabled:
        disabled.remove(flag)
    else:
        with subprocess.Popen([shutil.which('pylint'), file, '--disable=all', f'--enable={flag}']) as r:
            r.communicate()
            result = r.returncode
        if result:
            messages.append(f"Feel free to disable `{flag}` in `{file}`")

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

    files = [os.path.join(root,f) for root, dirs, filenames in os.walk(folder) for f in filenames if os.path.splitext(f)[1] == '.py']

    success = True

    messages = []

    for f in files:
        with open(f, encoding="utf-8") as myfile:
            lines = [l.replace(' ','') for l in myfile.readlines()]
        pylint_lines = [l.strip() for l in lines if l.startswith('#pylint:disable=')]
        disabled = set()
        for l in pylint_lines:
            disabled.update(l.split('=')[1].split(','))
        for r,d in accepted_pylint_commands.items():
            if r.match(f):
                for di in d:
                    disabled.discard(di)
        p = pathlib.Path(f)
        if p.parts[0] == 'tests':
            check_expected_pylint_disable(f, disabled, 'missing-function-docstring', messages)
            check_expected_pylint_disable(f, disabled, 'missing-module-docstring', messages)
            check_expected_pylint_disable(f, disabled, 'missing-class-docstring', messages)
            if p.parts[1] == 'epyccel':
                disabled.discard('reimported')
        if disabled:
            file_changed = f in diff
            disabled_str = ", ".join(f"`{d}`" for d in disabled)
            if file_changed:
                messages.append(f"[ERROR] New unexpected pylint disables found in `{f}`: {disabled_str}")
            else:
                messages.append(f"New unexpected pylint disables found in `{f}`: {disabled_str}")
            success &= (not file_changed)

    if messages:
        with open(args.output, mode='a', encoding="utf-8") as outfile:
            print("## Pylint Interaction", file=outfile)
            for m in messages:
                print(m, file=outfile)

    if not success:
        sys.exit(1)
