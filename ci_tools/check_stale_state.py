#pylint: disable=unused-import
""" Script to check up on a stale PR
"""
import argparse
import datetime
from git_evaluation_tools import check_previous_comments, leave_comment

stale_comment = "It looks like this PR may have been forgotten! Can you give us an update on its status please?\n\nIf you have got stuck then please don't struggle alone. Put a message in the [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb) and someone will try and help you out."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if any drafts look like they should be in review')
    parser.add_argument('pr_number', metavar='pr_number', type=str, nargs='+',
                            help='File where the markdown output will be printed')

    args = parser.parse_args()

    prs = args.pr_number

    for pr_id in prs:
        previous_comments, last_comment, last_date = check_previous_comments()
        if last_comment != stale_comment or (datetime.datetime.utcnow()-last_date)>datetime.timedelta(weeks=8):
            #TODO: Uncomment before merging and remove pylint disable
            #leave_comment(pr_id, stale_comment, True)
            pass
