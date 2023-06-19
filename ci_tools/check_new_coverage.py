""" Script to check that all new lines in the python files in the pyccel/ code folder are used in the tests
"""
import argparse
import json
import os
import shutil
import subprocess
from git_evaluation_tools import get_diff_as_json
from bot_tools.bot_funcs import Bot
import coverage_analysis_tools as cov

git = shutil.which('git')

def get_relevant_lines(diff, review):
    diff_hunk = review['diff_hunk']
    position = review['position']
    file = review['path']
    lines = diff[file]
    line_indicators = [(i, l) for i,l in enumerate(lines) if '@@' in l]
    line_key = next((i,l) for i,l in reversed(line_indicators) if i<position)
    offset = position-line_key[0]
    _, line_info, lines[0] = line_key[1].split('@@')
    line_info = line_info.strip()
    start_line = int(line_info.split(' ')[1].split(',')[0])
    return start_line + offset - 1

parser = argparse.ArgumentParser(description='Check that all new lines in the python files in the pyccel/ code folder are used in the tests')
parser.add_argument('diffFile', metavar='diffFile', type=str,
                        help='File containing the git diff output')
parser.add_argument('coverageFile', metavar='coverageFile', type=str,
                        help='File containing the coverage xml output')
parser.add_argument('output', metavar='output', type=str,
                        help='File where the markdown output will be printed')

args = parser.parse_args()

bot = Bot(pr_id = os.environ["PR_ID"], check_run_id = os.environ["CHECK_RUN_ID"], commit = os.environ['HEAD_SHA'])

current_diff = bot.get_diff()

#print("Diff:", current_diff)

revs = bot.get_bot_review_comments()

commented_lines = {(r[0]['path'], get_relevant_lines(current_diff, r[0])): r for r in revs.values()}

diff = get_diff_as_json(args.diffFile)
untested, file_contents = cov.get_untested_lines(args.coverageFile)

new_untested = cov.compare_coverage_to_diff(untested, diff)

new_untested = cov.allow_untested_error_calls(new_untested)

new_untested = cov.allow_untested_debug_code(new_untested)

old_comments, new_comments, existing_repeats = cov.get_json_summary(new_untested, file_contents, commented_lines)

for c,r in commented_lines:
    if c not in existing_repeats:
        print(r)
        bot.accept_coverage_fix(r)

success = cov.evaluate_success(bot, old_comments, new_comments, commented_lines)

cov.print_markdown_summary(old_comments + new_comments, os.environ['COMMIT'], args.output, bot.repo)

bot.post_coverage_review(new_comments, success)

bot.post_completed('success' if success else 'failure')

cov.show_results(new_untested)
