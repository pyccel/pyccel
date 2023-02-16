""" Script to process the output of numpydoc validator"
"""
import argparse
import sys

parser = argparse.ArgumentParser(description='Process the output of numpydoc validator')
parser.add_argument('report', metavar='report', type=str,
                        help='Report generated by numpydoc validator')
parser.add_argument('summary', metavar='summary', type=str,
                        help='Github step summary')
args = parser.parse_args()

error_codes = ['GL01', 'GL02', 'GL03', 'GL05', 'GL06', 'GL07', 'GL08', 'GL09',
               'GL10', 'SS01', 'SS02', 'SS03', 'SS04', 'SS05', 'SS06', 'ES01',
               'PR01', 'PR02', 'PR03', 'PR04', 'PR05', 'PR06', 'PR07', 'PR08',
               'PR09', 'PR10', 'RT01', 'RT02', 'RT03', 'RT04', 'RT05', 'YD01',
               'SA02', 'SA03', 'SA04']

warning_codes = ['EX01', 'SA01']

errors = {}
warnings = {}
parsing_errors = []
with open(args.report, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            file_name, code, msg = line.split(':', maxsplit=2)
            if code in error_codes:
                if file_name not in errors:
                    errors[file_name] = [msg]
                else:
                    errors[file_name].append(msg)
                parsing_errors.append(line)
            elif code in warning_codes:
                if file_name not in warnings:
                    warnings[file_name] = [msg]
                else:
                    warnings[file_name].append(msg)
            else:
                parsing_errors.append(line)
        # This catch is for errors that arise from the line split
        # when the line does not follow the file:err_code_msg pattern
        # ie, in the case of parsing errors.
        except ValueError:
            parsing_errors.append(line)

fail = len(errors) > 0 or len(parsing_errors) > 0

with open(args.summary, 'a', encoding='utf-8') as f:
    print('# Part 2 : Numpydoc Validation:', file=f)
    print(f'## {"FAILURE" if fail else "SUCCESS"}!', file=f)
    if fail:
        print('### ERRORS!', file=f)
    for file_name, errs in errors.items():
        print(f'#### {file_name}', file=f)
        print(''.join(f'- {err}' for err in errs), file=f)
    if (len(warnings) > 0):
        print('### WARNINGS!', file=f)
    for file_name, warns in warnings.items():
        print(f'#### {file_name}', file=f)
        print(''.join(f'- {warn}' for warn in warns), file=f)
    if (len(parsing_errors) > 0):
        print('### PARSING ERRORS!', file=f)
        parsing_errors = ['\n' if 'warn(msg)' in err else err for err in parsing_errors]
        print(''.join(f'{add_warn}' for add_warn in parsing_errors), file=f)

sys.exit(fail)

