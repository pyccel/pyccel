""" Script to check that all new lines in the python files in the pyccel/ code folder are used in the tests
"""
import json
import os
import argparse
from git_evaluation_tools import get_diff_as_json
from bot_tools.bot_funcs import Bot
import coverage_analysis_tools as cov

def get_relevant_lines(diff_hunk, position):
    lines = diff_hunk.split('\n')
    _, line_info, lines[0] = lines[0].split('@@')
    print(lines)
    print(line_info)
    start_line, old_len, new_start_line, new_len = line_info.split(',')
    return new_start_line + position - 1

parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
parser.add_argument('diffFile', metavar='diffFile', type=str,
                        help='File containing the git diff output')
parser.add_argument('coverageFile', metavar='coverageFile', type=str,
                        help='File containing the coverage xml output')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

diff = get_diff_as_json(args.diffFile)
untested, file_contents = cov.get_untested_lines(args.coverageFile)

new_untested = cov.compare_coverage_to_diff(untested, diff)

new_untested = cov.allow_untested_error_calls(new_untested)

new_untested = cov.allow_untested_debug_code(new_untested)

comments = cov.get_json_summary(new_untested, file_contents)

bot = Bot(pr_id = os.environ["PR_ID"], check_run_id = os.environ["CHECK_RUN_ID"], commit = os.environ['HEAD_SHA'])

revs = bot.get_bot_review_comments()

print(revs)

handled_lines = [get_relevant_lines(r['diff_hunk'], r['original_position']) for r in revs]

cov.print_markdown_summary(comments, os.environ['COMMIT'], args.output, bot.repo)

bot.post_coverage_review(comments)

bot.post_completed('failure' if comments else 'success')

cov.show_results(new_untested)
