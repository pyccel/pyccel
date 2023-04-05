""" Script to check that Pyccel coding conventions concerning Pylint are correctly followed
"""
import argparse
import os
import pathlib
import re
import subprocess

accepted_pylint_commands = {re.compile('.*/IMPORTING_EXISTING_IDENTIFIED3.py'):['reimported'],
                            re.compile('.*/TOO_FEW_ARGS.py'):['no-value-for-parameter'],
                            re.compile('.*/UNKNOWN_IMPORT.py'):['unused-import'],
                            re.compile('.*/UNKNOWN_IMPORT2.py'):['unused-import'],
                            re.compile('.*/USELESS_EXPRESSION.py'):['pointless-statement'],
                            re.compile('./tests/errors/known_bugs/dicts.py'):['pointless-statement'],
                            re.compile('.*/syntax/.*'):['pointless-statement','undefined-variable'],
                            re.compile('./tests/codegen/fcode/scripts/precision.py'):['unused-variable']}

def check_expected_pylint_disable(file, disabled, flag, outfile):
    """
    Check for an expected pylint disable flag.

    Check for an expected pylint disable flag. If the flag is present
    then it is ignored by removing it from the list. Otherwise if the
    file raises the error a message is printed to the outfile recommending
    that the flag be disabled in the file.

    Parameters
    ----------
    file : file object
        The file being analysed.
    disabled : list
        The name of all pylint flags disabled in this file.
    flag : str
        The name of the flag being investigated.
    outfile : file object
        The file where any messages should be printed.
    """
    if flag in disabled:
        disabled.remove(flag)
    else:
        with subprocess.Popen(['pylint', file, '--disable=all', f'--enable={flag}']) as r:
            r.communicate()
            result = r.returncode
        if result:
            print(f"Feel free to disable {flag} in {file}", file=outfile)

parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
parser.add_argument('folder', type=str,
                        help='The folder to be analysed')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

folder = args.folder

files = [os.path.join(root,f) for root, dirs, filenames in os.walk(folder) for f in filenames if os.path.splitext(f)[1] == '.py']

with open(args.output, mode='a', encoding="utf-8") as outfile:
    for f in files:
        with open(f, encoding="utf-8") as myfile:
            lines = [l.replace(' ','') for l in myfile.readlines()]
        pylint_lines = [l.strip() for l in lines if l.startswith('#pylint:disable=')]
        disabled = []
        for l in pylint_lines:
            disabled.extend(l.split('=')[1].split(','))
        for r,d in accepted_pylint_commands.items():
            if r.match(f):
                for di in d:
                    try:
                        disabled.remove(di)
                    except ValueError:
                        pass
        p = pathlib.Path(f)
        if p.parts[0] == 'tests':
            check_expected_pylint_disable(f, disabled, 'missing-function-docstring', outfile)
            check_expected_pylint_disable(f, disabled, 'missing-module-docstring', outfile)
            check_expected_pylint_disable(f, disabled, 'missing-class-docstring', outfile)
            if p.parts[1] == 'epyccel':
                try:
                    disabled.remove('reimported')
                except ValueError:
                    pass
        if disabled:
            print(f"Unexpected pylint disables found in {f}:", disabled, file=outfile)
