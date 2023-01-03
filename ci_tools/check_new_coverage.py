""" Script to check that all new lines in the python files in the pyccel/ code folder are used in the tests
"""
import json
import argparse
from git_evaluation_tools import get_diff_as_json
import coverage_analysis_tools as cov

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

with open(args.gitEvent, encoding="utf-8") as pr_data_file:
    pr_data = json.load(pr_data_file)

cov.print_markdown_summary(new_untested, file_contents, pr_data["pull_request"]["base"]["sha"], args.output)

cov.show_results(new_untested)

