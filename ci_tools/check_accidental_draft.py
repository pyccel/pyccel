""" Script to check if a dev has forgotten to remove the draft status from their PR.
"""
import argparse
from check_stale_state import stale_comment
from git_evaluation_tools import check_previous_comments, leave_comment

draft_comment = "Your PR is passing all the tests but it is still marked as draft!\n\nDid you forget to remove the draft status? If you are ready to begin the review process then you can go ahead and remove it :rocket:\n\nIf you have got stuck and need a hand finishing your PR don't hesitate to ask the devs on the [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb)."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if any drafts look like they should be in review')
    parser.add_argument('pr_number', metavar='pr_number', type=str, nargs='+',
                            help='File where the markdown output will be printed')

    args = parser.parse_args()

    prs = args.pr_number

    for pr_id in prs:
        previous_comments, last_comment, last_date = check_previous_comments()
        if last_comment not in (draft_comment, stale_comment):
            #TODO: Uncomment before merging
            #leave_comment(pr_id, draft_comment, True)
            pass
