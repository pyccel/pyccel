""" Script to check that all new lines in the python files in the pyccel/ code folder are used in the tests
"""
import argparse
import os
from git_evaluation_tools import get_diff_as_json
from bot_tools.bot_funcs import Bot
import coverage_analysis_tools as cov

def get_relevant_line(diff, review):
    """
    Get the line of the comment in the current version of the code.

    From a review on the code and the current diff shown on the PR
    for the branch, calculate the line of the comment in the code
    currently on the branch.

    Parameters
    ----------
    diff : dict
        A dictionary whose keys are files and whose comments contain
        the diff as shown on the PR starting from the first code
        blob.

    review : dict
        A dictionary describing the review comment which was left on
        the blob of code.

    Returns
    -------
    int
        The relevant line of code in the current commit.
    """
    # Get updated position according to git
    position = review['position']

    # Get code
    file = review['path']
    lines = diff[file]
    print(file, lines)

    # Get line numbers indicated by the blob
    line_indicators = [(i, l) for i,l in enumerate(lines) if '@@' in l]
    print("looking for lines:", position, line_indicators)

    # Find the relevant blob
    line_key = next((i,l) for i,l in reversed(line_indicators) if i<position)

    # Calculate the new line number
    offset = position-line_key[0]
    _, line_info, _ = line_key[1].split('@@')
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
print(current_diff.keys())

revs = bot.get_bot_review_comments()

print("Review comments : ", len(revs))

commented_lines = {(r[0]['path'], get_relevant_line(current_diff, r[0])): r for r in revs}

diff = get_diff_as_json(args.diffFile)
untested, file_contents = cov.get_untested_lines(args.coverageFile)

new_untested = cov.compare_coverage_to_diff(untested, diff)

new_untested = cov.allow_untested_error_calls(new_untested)

new_untested = cov.allow_untested_debug_code(new_untested)

print("Commented lines : ")
print(commented_lines)
for (p, l), r in commented_lines.items():
    print(p,l,r)
print("--------------------------------------------------------")

old_comments, new_comments = cov.get_json_summary(new_untested, file_contents, commented_lines)

print("Discovered:")
print(old_comments)
print(new_comments)
print("--------------------------------------------------------")

success = cov.evaluate_success(old_comments, new_comments, commented_lines)

cov.print_markdown_summary(old_comments + new_comments, os.environ['COMMIT'], args.output, bot.repo)

bot.post_coverage_review(new_comments, success)

bot.post_completed('success' if success else 'failure')

cov.show_results(new_untested)
