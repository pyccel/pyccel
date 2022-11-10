from git_evaluation_tools import get_diff_as_json
import compare_coverage_to_diff as cov
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
parser.add_argument('diffFile', metavar='diffFile', type=str,
                        help='File containing the git diff output')
parser.add_argument('coverageFile', metavar='coverageFile', type=str,
                        help='File containing the coverage xml output')
parser.add_argument('gitEvent', metavar='gitEvent', type=str,
                        help='File containing the json description of the triggering event')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

diff = get_diff_as_json(args.diffFile)
untested, file_contents = cov.get_untested_lines(args.coverageFile)

new_untested = cov.allow_untested_error_calls(cov.compare_coverage_to_diff(untested, diff))

pr_data = json.load(open(args.gitEvent))

cov.print_markdown_summary(new_untested, file_contents, pr_data["after"], args.output)

cov.check_results(new_untested)

